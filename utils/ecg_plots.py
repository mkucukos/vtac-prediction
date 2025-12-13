import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from datetime import timedelta
from scipy.signal import medfilt

# ================================
# A) STANDARDIZATION (NO PLOTTING)
# ================================
def compute_refs_and_zscores(results_df, sampling_rate=250, window_len_sec=30):
    """
    REAL-TIME / CAUSAL STANDARDIZATION (INDEX-BASED)

    - No baseline
    - No VTAC logic (except label)
    - Wait for at least 12 prior windows (points), NOT time
    - TMV_Global & QRS_Global use evolving references
    - Z-scores computed using past windows only
    """

    df = results_df.copy()
    df = df.sort_values(["Record", "Start"]).reset_index(drop=True)

    min_history = 12  # <<< NEW

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
        "QRS_Duration",
        "QRS_Area",
        "QRS_Skewness",
        "ST_Deviation_Mean",
        "ST_Slope_Mean",
        "AC_ECG_Peak",
        "AC_ECG_Lag_Sec",
        "AC_ECG_MeanAroundPeak",
        "AC_RR_Peak",
        "AC_RR_Lag_Beats",
        "AC_RR_MeanAroundPeak",
    ]

    df["TMV_Global"] = np.nan
    df["QRS_Global"] = np.nan
    df["VTAC_Label"] = 0

    for field in z_fields + ["TMV_Global", "QRS_Global"]:
        df[f"{field}_Z"] = np.nan

    # -----------------------------
    # Process per Record
    # -----------------------------
    for record_id, g in df.groupby("Record"):

        g = g.sort_values("Start").reset_index()
        idx = g["index"].values

        past_twaves = []
        past_qrs = []

        # ---------- VTAC labeling (non-causal OK) ----------
        vtac_times = g.loc[
            g["Label"].astype(str).str.upper() == "VTAC", "Start"
        ].values

        if len(vtac_times) > 0:
            vtac_start = float(vtac_times.min())
            vtac_end = float(vtac_times.max() + window_len_sec)

            df.loc[
                (df["Record"] == record_id)
                & (df["Start"] >= vtac_start)
                & (df["Start"] <= vtac_end),
                "VTAC_Label",
            ] = 1

        # -----------------------------
        # Window loop
        # -----------------------------
        for i, row in g.iterrows():

            # ---------- TMV reference ----------
            tw = row.get("T_Wave")
            if isinstance(tw, (list, np.ndarray)) and len(tw) == 100:
                tw = np.asarray(tw, dtype=float)

                if len(past_twaves) >= min_history:
                    ref_twave = np.median(np.vstack(past_twaves), axis=0)
                    df.at[idx[i], "TMV_Global"] = float(
                        np.mean((tw - ref_twave) ** 2)
                    )

                past_twaves.append(tw)

            # ---------- QRS reference ----------
            qrs = row.get("QRS_Wave")
            if isinstance(qrs, (list, np.ndarray)) and len(qrs) == 100:
                qrs = np.asarray(qrs, dtype=float)

                if len(past_qrs) >= min_history:
                    ref_qrs = np.median(np.vstack(past_qrs), axis=0)
                    df.at[idx[i], "QRS_Global"] = float(
                        np.mean((qrs - ref_qrs) ** 2)
                    )

                past_qrs.append(qrs)

            # ---------- Z-scoring ----------
            if i < min_history:
                continue  # warm-up only

            past_idx = idx[:i]

            for field in z_fields + ["TMV_Global", "QRS_Global"]:

                current_val = df.at[idx[i], field]
                if pd.isna(current_val):
                    continue

                past_values = df.loc[past_idx, field].dropna()
                if len(past_values) < min_history:
                    continue

                median = float(np.nanmedian(past_values))
                q25, q75 = np.nanpercentile(past_values, [25, 75])
                iqr = q75 - q25

                if iqr < 1e-6:
                    continue

                z = (current_val - median) / (iqr / 1.349 + 1e-6)
                df.at[idx[i], f"{field}_Z"] = np.clip(z, -10, 10)

    return df

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
    smooth_kernel=3,   # <-- NEW
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

        # --- Median filter smoothing (temporal) ---
        if smooth_kernel and smooth_kernel > 1:
            record_df["Prob_RF"] = medfilt(record_df["Prob_RF"].values, kernel_size=smooth_kernel)
            record_df["Reg_RF"]  = medfilt(record_df["Reg_RF"].values,  kernel_size=smooth_kernel)

            # Optional: if you later re-enable XGB plots
            record_df["Prob_XGB"] = medfilt(record_df["Prob_XGB"].values, kernel_size=smooth_kernel)
            record_df["Reg_XGB"]  = medfilt(record_df["Reg_XGB"].values,  kernel_size=smooth_kernel)


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