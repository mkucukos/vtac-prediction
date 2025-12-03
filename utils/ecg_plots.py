import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from datetime import timedelta

# ================================
# A) STANDARDIZATION (NO PLOTTING)
# ================================
def compute_refs_and_zscores(results_df, sampling_rate=250, window_len_sec=30):
    """
    Runs the exact same baseline selection, ref building, TMV_Global/QRS_Global,
    z-scoring, and VTAC_Label logic as your original function, but returns the
    concatenated DataFrame WITHOUT plotting.
    """
    unique_records = sorted(results_df["Record"].unique())
    all_zscore_dfs = []

    for record_id in unique_records:

        # Pull only this record
        record_df = results_df[results_df["Record"] == record_id].reset_index(drop=True)
        if record_df.empty:
            continue

        # Ensure Start is numeric seconds
        if not np.issubdtype(record_df["Start"].dtype, np.number):
            record_df["Start"] = pd.to_datetime(record_df["Start"])
            record_df["Start"] = (
                record_df["Start"] - record_df["Start"].iloc[0]
            ).dt.total_seconds()

        # Extract VTAC window start times
        vtac_rows = record_df[record_df["Label"].astype(str).str.upper() == "VTAC"]
        vtac_times = vtac_rows["Start"].values if not vtac_rows.empty else []

        # ------------------------------------------------------------
        # NEW BASELINE SELECTION (Case 1–3 accepted; Case 4–5 skipped)
        # ------------------------------------------------------------
        vtac_start = float(vtac_times.min()) if len(vtac_times) > 0 else None

        # If no VTAC for this record → skip
        if vtac_start is None:
            print(f"[SKIP] Record {record_id}: No VTAC detected.")
            continue

        # Threshold format: (required_pre_vtac_seconds, baseline_cutoff_seconds)
        valid_thresholds = [
            (500, 300),   # Case 1 → ≥200 sec baseline
            (400, 240),   # Case 2 → ≥160 sec baseline
            (300, 180),   # Case 3 → ≥120 sec baseline
        ]

        invalid_thresholds = [
            (200, 120),   # Case 4 → too short (80 sec)
            (100,  60),   # Case 5 → too short (40 sec)
        ]

        baseline_mask = None

        # ---------- Try valid baseline ranges ----------
        for required_pre, cutoff in valid_thresholds:
            if (record_df["Start"] < (vtac_start - required_pre)).any():
                baseline_mask = record_df["Start"] < (vtac_start - cutoff)
                break

        # ---------- If no valid baseline found, evaluate invalid thresholds ----------
        if baseline_mask is None:

            # Case 4–5 → skip immediately
            for required_pre, cutoff in invalid_thresholds:
                if (record_df["Start"] < (vtac_start - required_pre)).any():
                    print(
                        f"[SKIP] Record {record_id}: Only {required_pre}s pre-VTAC "
                        f"(minimum required is 300/400/500)."
                    )
                    break
            else:
                # No baseline at all → skip
                print(f"[SKIP] Record {record_id}: No usable pre-VTAC baseline.")
            
            continue  # ⚠️ Skip record

        # ----------- From here baseline_mask is valid -----------

        # ---- Subject-specific T-wave reference (median, len 100) ----
        baseline_twaves = [
            np.array(row["T_Wave"], dtype=float)
            for _, row in record_df[baseline_mask].iterrows()
            if isinstance(row.get("T_Wave"), list) and len(row["T_Wave"]) == 100
        ]
        if len(baseline_twaves) < 5:
            continue
        ref_twave = np.median(np.vstack(baseline_twaves), axis=0)

        # ---- Subject-specific QRS reference (median, len 100) ----
        baseline_qrs = [
            np.array(row["QRS_Wave"], dtype=float)
            for _, row in record_df[baseline_mask].iterrows()
            if isinstance(row.get("QRS_Wave"), list) and len(row["QRS_Wave"]) == 100
        ]
        ref_qrs = (
            np.median(np.vstack(baseline_qrs), axis=0)
            if len(baseline_qrs) >= 5
            else None
        )

        # ---- TMV_Global (MSE vs ref_twave) ----
        record_df["TMV_Global"] = np.nan
        for i, row in record_df.iterrows():
            tw = row.get("T_Wave")
            if isinstance(tw, list) and len(tw) == 100:
                record_df.at[i, "TMV_Global"] = float(
                    np.mean((np.array(tw, dtype=float) - ref_twave) ** 2)
                )

        # ---- QRS_Global (MSE vs ref_qrs, enforce len 100) ----
        record_df["QRS_Global"] = np.nan
        if ref_qrs is not None:
            for i, row in record_df.iterrows():
                qrs = row.get("QRS_Wave")
                if isinstance(qrs, list) and len(qrs) == 100:
                    record_df.at[i, "QRS_Global"] = float(
                        np.mean((np.array(qrs, dtype=float) - ref_qrs) ** 2)
                    )

        # ---- Fields to Z-score (baseline-only) ----
        z_fields = [
            "TMV_Score",
            "QT_Interval",
            "Mean_HR",
            "Max_HR",
            "Min_HR",
            "RMSSD",
            "SDNN",
            "T_Flatness",
            "TWAmp_Std",
            "TWAmp_CV",
            "TMV_Global",
            "QRS_Duration",
            "QRS_Area",
            "QRS_Skewness",
            "ST_Deviation_Mean",
            "ST_Slope_Mean",
            "QRS_Global",
            "AC_ECG_Peak",
            "AC_ECG_Lag_Sec",
            "AC_ECG_MeanAroundPeak",
            "AC_RR_Peak",
            "AC_RR_Lag_Beats",
            "AC_RR_MeanAroundPeak",
        ]
        if "QRS_Shape_Var" in record_df.columns:
            z_fields.append("QRS_Shape_Var")

        for field in z_fields:
            if field in record_df.columns:
                values = record_df.loc[baseline_mask, field].dropna()
                z_col = f"{field}_Z"
                record_df[z_col] = np.nan
                if len(values) >= 5:
                    median = float(np.nanmedian(values))
                    std = float(np.nanstd(values))
                    record_df[z_col] = record_df[field].apply(
                        lambda x: (x - median) / (std + 1e-6) if pd.notna(x) else np.nan
                    )
                    record_df[z_col] = record_df[z_col].fillna(10).clip(upper=10)

        # ---- VTAC labeling & truncation ----
        record_df["VTAC_Label"] = 0
        if len(vtac_times) > 0:
            vtac_start = float(vtac_times.min())
            vtac_end = float(vtac_times.max() + window_len_sec)
            record_df.loc[
                (record_df["Start"] >= vtac_start) & (record_df["Start"] <= vtac_end),
                "VTAC_Label",
            ] = 1
            record_df = record_df[record_df["Start"] <= vtac_end].reset_index(drop=True)
        else:
            vtac_end = None

        all_zscore_dfs.append(record_df)

    if not all_zscore_dfs:
        return pd.DataFrame()
    return pd.concat(all_zscore_dfs, ignore_index=True)

# ==================
# B) PLOTTING ONLY
# ==================
def plot_subject_panels(processed_df, sampling_rate=250, window_len_sec=30):
    """
    Plots from a DataFrame already processed by compute_refs_and_zscores().
    Uses the same fields, thresholds, and visuals as your original function.
    """
    # This z_fields list must match what was used in the compute step
    z_fields = [
        "TMV_Score",
        "QT_Interval",
        "Mean_HR",
        "Max_HR",
        "Min_HR",
        "RMSSD",
        "SDNN",
        "T_Flatness",
        "TWAmp_Std",
        "TWAmp_CV",
        "TMV_Global",
        "QRS_Duration",
        "QRS_Area",
        "QRS_Skewness",
        "ST_Deviation_Mean",
        "ST_Slope_Mean",
        "QRS_Global",
        "AC_ECG_Peak",
        "AC_ECG_Lag_Sec",
        "AC_ECG_MeanAroundPeak",
        "AC_RR_Peak",
        "AC_RR_Lag_Beats",
        "AC_RR_MeanAroundPeak",
    ]
    if "QRS_Shape_Var" in processed_df.columns:
        z_fields.append("QRS_Shape_Var")

    unique_records = sorted(processed_df["Record"].unique())

    for record_id in unique_records:
        record_df = processed_df[processed_df["Record"] == record_id].reset_index(
            drop=True
        )
        if record_df.empty:
            continue

        # plotting uses the same time base
        window_times = record_df["Start"] / sampling_rate

        fig, axes = plt.subplots(
            len(z_fields) + 1, 1, figsize=(14, 3 * (len(z_fields) + 1)), sharex=True
        )

        # Raw ECG overlay
        for i, row in record_df.iterrows():
            ecg_raw = row.get("ECG_Raw")
            if isinstance(ecg_raw, list) and len(ecg_raw) > 0:
                t = np.linspace(
                    window_times.iloc[i],
                    window_times.iloc[i] + window_len_sec,
                    len(ecg_raw),
                )
                axes[0].plot(t, ecg_raw, alpha=0.5)
        axes[0].set_ylabel("Raw ECG")
        axes[0].set_title(f"Record: {record_id} | Z-Scored Features")

        def plot_z(ax, series, label, color=None, threshold=1.5):
            filled_series = series.fillna(3)
            # Drop duplicate x by keeping first (prevents vertical lines)
            x_all = window_times.round(6)
            keep_mask = ~x_all.duplicated(keep="first")
            x = window_times[keep_mask]
            y = filled_series[keep_mask]

            ax.plot(x, y, label=label, color=color)
            for i in range(len(y)):
                if y.iloc[i] > threshold:
                    start = x.iloc[i]
                    end = start + window_len_sec
                    ax.axvspan(start, end, color="red", alpha=0.2)
            ax.axhline(y=threshold, color="gray", linestyle="--", linewidth=1)
            ax.legend()

        z_plot_fields = [f"{f}_Z" for f in z_fields]
        # FIX: Ensure colors list has enough colors for all z_fields
        colors = [
            "blue",
            "navy",
            "purple",
            "red",
            "darkred",
            "salmon",
            "green",
            "darkgreen",
            "slateblue",
            "orange",
            "darkorange",
            "magenta",
            "teal",
            "brown",
            "olive",
            "gold",
            "cyan",
            "pink",
            "gray",
            "lime",
            "maroon",
            "coral",
            "indigo",  # Added more colors
        ]

        for i, (z_field, color) in enumerate(zip(z_plot_fields, colors), start=1):
            if z_field in record_df.columns:
                plot_z(axes[i], record_df[z_field], z_field, color)
                axes[i].set_ylabel(z_field)

        # VTAC markers (if present in processed_df)
        vtac_rows = record_df[record_df["VTAC_Label"] == 1]
        if not vtac_rows.empty:
            vtac_start = vtac_rows["Start"].min()
            vtac_end = vtac_rows["Start"].max() + window_len_sec
            for ax in axes:
                ax.axvline(
                    x=vtac_start / sampling_rate,
                    color="black",
                    linestyle="--",
                    alpha=0.8,
                    label="VTAC Start",
                )
                ax.axvline(
                    x=vtac_end / sampling_rate,
                    color="blue",
                    linestyle="--",
                    alpha=0.8,
                    label="VTAC End",
                )
            handles, labels = axes[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes[0].legend(by_label.values(), by_label.keys())

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()



def plot_standardized_qt_tmv_with_dual_prediction(
    results_df,
    alarms_df,
    features,
    rf_model_path="random_forest_vtac_model.joblib",
    xgb_model_path="xgboost_vtac_model.joblib",
    rf_model_path_reg="random_forest_vtac_model_regression.joblib",
    xgb_model_path_reg="xgboost_vtac_model_regression.joblib",
    z_threshold=1.5,
    output_dir="plots",
):
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    rf_model = joblib.load(rf_model_path)
    xgb_model = joblib.load(xgb_model_path)
    rf_model_reg = joblib.load(rf_model_path_reg)
    xgb_model_reg = joblib.load(xgb_model_path_reg)

    unique_records = sorted(results_df["Record"].unique())

    for record_id in unique_records:
        print(f"\n[INFO] Processing record: {record_id}")

        # Extract rows for this record
        record_df = results_df[results_df["Record"] == record_id].reset_index(drop=True)

        missing_features = [f for f in features if f not in record_df.columns]
        if record_df.empty or missing_features:
            print(f"[SKIP] {record_id}: Missing features or empty DF.")
            continue

        # Ensure Start is datetime
        record_df["Start"] = pd.to_datetime(record_df["Start"], errors="coerce")
        if record_df["Start"].isna().all():
            print(f"[SKIP] {record_id}: All Start times are NaT.")
            continue

        # Model Predictions
        X = record_df[features]
        record_df["Prob_RF"] = rf_model.predict_proba(X)[:, 1]
        record_df["Prob_XGB"] = xgb_model.predict_proba(X)[:, 1]
        record_df["Reg_RF"] = rf_model_reg.predict(X)
        record_df["Reg_XGB"] = xgb_model_reg.predict(X)

        # Keep only rows with valid predictions
        valid_mask = (
            X.notna().all(axis=1) &
            record_df["Prob_RF"].notna() &
            record_df["Prob_XGB"].notna() &
            record_df["Reg_RF"].notna() &
            record_df["Reg_XGB"].notna()
        )
        record_df = record_df[valid_mask].reset_index(drop=True)

        if record_df.empty:
            print(f"[SKIP] {record_id}: No valid prediction windows.")
            continue

        # Recompute time axis
        window_times = (record_df["Start"] - record_df["Start"].iloc[0]).dt.total_seconds()

        # VTAC Alarm Times
        record_alarms = alarms_df[alarms_df["Files"] == record_id]
        record_start = record_df["Start"].iloc[0]

        # Z-score fields (dynamic)
        z_fields = [
            ("TMV_Score_Z", "darkblue", "TMV Score (Z)"),
            ("QT_Interval_Z", "purple", "QT Interval (Z)"),
            ("Mean_HR_Z", "green", "Mean HR (Z)"),
            ("TMV_Global_Z", "crimson", "TMV Global (Z)"),
            ("QRS_Duration_Z", "slateblue", "QRS Duration (Z)"),
            ("AC_ECG_Peak_Z", "crimson", "AC Peak (Z)"),
        ]

        num_z = len(z_fields)
        total_plots = 1 + num_z + 2  # 1 raw + Z features + 2 prediction plots

        # Create dynamic subplot count
        fig, axes = plt.subplots(total_plots, 1, figsize=(40, 2.2 * total_plots), sharex=True)

        # ----- RAW ECG -----
        for i, row in record_df.iterrows():
            ecg_raw = row.get("ECG_Raw", None)
            if isinstance(ecg_raw, list) and len(ecg_raw) > 0:
                t = np.linspace(window_times[i], window_times[i] + 30, len(ecg_raw))
                axes[0].plot(t, ecg_raw, alpha=0.5)

        axes[0].set_ylabel("Raw ECG")
        axes[0].set_ylim(-500, 500)
        axes[0].set_title(f"Record: {record_id}")

        # ---- Helper to shade Z-threshold violations ----
        def shade_z(ax, vals, times, color):
            for i, z in enumerate(vals):
                if pd.isna(z): continue
                if z > z_threshold:
                    ax.axvspan(times[i], times[i] + 30, color=color, alpha=0.2)
                elif z < -z_threshold:
                    ax.axvspan(times[i], times[i] + 30, color="blue", alpha=0.2)

        # ----- Z-score plots -----
        for j, (field, color, label) in enumerate(z_fields):
            ax = axes[j + 1]
            ax.plot(window_times, record_df[field], color=color, label=label)
            ax.axhline(z_threshold, color="red", linestyle="--")
            ax.axhline(-z_threshold, color="blue", linestyle="--")
            shade_z(ax, record_df[field], window_times, color)
            ax.set_ylabel(label)
            ax.legend()

        # ----- Classification Probabilities -----
        prob_ax = axes[1 + num_z]
        prob_ax.plot(window_times, record_df["Prob_RF"], color="orange", label="RF Prob")
        prob_ax.plot(window_times, record_df["Prob_XGB"], color="red", linestyle="--", label="XGB Prob")
        prob_ax.set_ylim(0, 1)
        prob_ax.set_title("Predicted VTAC Probability")
        prob_ax.legend()

        # ----- Regression Risk -----
        reg_ax = axes[2 + num_z]
        reg_ax.plot(window_times, record_df["Reg_RF"], color="orange", label="RF Reg")
        reg_ax.plot(window_times, record_df["Reg_XGB"], color="red", linestyle="--", label="XGB Reg")
        reg_ax.set_ylim(0, 1)
        reg_ax.set_title("Predicted VTAC Risk")
        reg_ax.set_xlabel("Time (s)")
        reg_ax.legend()

        # ----- Add VTAC markers to all axes -----
        max_time = window_times.max() + 30

        for ax in axes:
            ax.set_xlim(0, max_time)
            for _, alarm in record_alarms.iterrows():
                vt_start = pd.to_datetime(alarm["StartTime"], errors="coerce")
                if pd.isna(vt_start): continue
                vt_end = vt_start + timedelta(seconds=int(alarm["Duration"]))
                s = (vt_start - record_start).total_seconds()
                e = (vt_end - record_start).total_seconds()
                ax.axvline(s, color="black", linestyle="--")
                ax.axvline(e, color="black", linestyle="--")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{record_id}_dual_prediction_plot.png")
        # plt.savefig(save_path, dpi=300)
        plt.show()

        print(f"[SAVED] {save_path}")