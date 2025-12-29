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
def compute_refs_and_zscores(
    results_df,
    sampling_rate=250,
    window_len_sec=30,
    min_history=12
):
    """
    REAL-TIME / CAUSAL STANDARDIZATION (INDEX-BASED)

    - No baseline
    - No VTAC logic (except label)
    - Wait for at least min_history prior windows (points), NOT time
    - TMV_Global & QRS_Global use evolving references
    - Z-scores computed using past windows only
    """

    df = results_df.copy()
    df = df.sort_values(["Record", "Start"]).reset_index(drop=True)

    z_fields = [
        "TMV_Score", "QT_Interval",
        "Mean_HR", "Max_HR", "Min_HR",
        "RMSSD", "SDNN",
        "T_Flatness", "TWAmp_Std", "TWAmp_CV",
        "QRS_Duration", "QRS_Area", "QRS_Skewness",
        "ST_Deviation_Mean", "ST_Slope_Mean",
        "AC_ECG_Peak", "AC_ECG_Lag_Sec", "AC_ECG_MeanAroundPeak",
        "AC_RR_Peak", "AC_RR_Lag_Beats", "AC_RR_MeanAroundPeak",
    ]

    # Global / derived fields
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
        ]

        if not vtac_times.empty:
            vtac_start = vtac_times.min()
            vtac_max = vtac_times.max()

            # ---- FIX: handle datetime vs sample index safely ----
            if isinstance(vtac_max, (pd.Timestamp, np.datetime64)):
                vtac_end = vtac_max + pd.to_timedelta(window_len_sec, unit="s")
            else:
                vtac_end = vtac_max + int(window_len_sec * sampling_rate)

            df.loc[
                (df["Record"] == record_id)
                & (df["Start"] >= vtac_start)
                & (df["Start"] <= vtac_end),
                "VTAC_Label",
            ] = 1

        # -----------------------------
        # Window loop (causal)
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
                continue  # warm-up only (causal)

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

                robust_std = iqr / 1.349
                z = (current_val - median) / (robust_std + 1e-6)

                df.at[idx[i], f"{field}_Z"] = np.clip(z, -10, 10)

    return df


# ==================
# B) PLOTTING ONLY
# ==================
def compute_refs_and_zscores(
    results_df,
    sampling_rate=250,
    window_len_sec=30,
    min_history=12,
    nan_frac_threshold=0.5,
):

    df = results_df.copy()
    df = df.sort_values(["Record", "Start"]).reset_index(drop=True)

    z_fields = [
        "TMV_Score", "QT_Interval",
        "Mean_HR", "Max_HR", "Min_HR",
        "RMSSD", "SDNN",
        "T_Flatness", "TWAmp_Std", "TWAmp_CV",
        "QRS_Duration", "QRS_Area", "QRS_Skewness",
        "ST_Deviation_Mean", "ST_Slope_Mean",
        "AC_ECG_Peak", "AC_ECG_Lag_Sec", "AC_ECG_MeanAroundPeak",
        "AC_RR_Peak", "AC_RR_Lag_Beats", "AC_RR_MeanAroundPeak",
    ]

    df["TMV_Global"] = np.nan
    df["QRS_Global"] = np.nan
    df["VTAC_Label"] = 0
    df["avail"] = 0

    for field in z_fields + ["TMV_Global", "QRS_Global"]:
        df[f"{field}_Z"] = np.nan

    z_cols = [f"{f}_Z" for f in z_fields + ["TMV_Global", "QRS_Global"]]

    # make dataframe numeric once
    df[z_cols] = df[z_cols].apply(pd.to_numeric, errors="coerce")

    for record_id, g in df.groupby("Record"):
        g = g.sort_values("Start").reset_index()
        idx = g["index"].values

        past_twaves = []
        past_qrs = []

        vtac_times = g.loc[
            g["Label"].astype(str).str.upper() == "VTAC", "Start"
        ]

        if not vtac_times.empty:
            vtac_start = vtac_times.min()
            vtac_max = vtac_times.max()

            if isinstance(vtac_max, (pd.Timestamp, np.datetime64)):
                vtac_end = vtac_max + pd.to_timedelta(window_len_sec, unit="s")
            else:
                vtac_end = vtac_max + int(window_len_sec * sampling_rate)

            df.loc[
                (df["Record"] == record_id)
                & (df["Start"] >= vtac_start)
                & (df["Start"] <= vtac_end),
                "VTAC_Label",
            ] = 1

        for i, row in g.iterrows():

            # ---- references ----
            tw = row.get("T_Wave")
            if isinstance(tw, (list, np.ndarray)) and len(tw) == 100:
                tw = np.asarray(tw, dtype=float)
                if len(past_twaves) >= min_history:
                    ref = np.median(np.vstack(past_twaves), axis=0)
                    df.at[idx[i], "TMV_Global"] = float(np.mean((tw - ref) ** 2))
                past_twaves.append(tw)

            qrs = row.get("QRS_Wave")
            if isinstance(qrs, (list, np.ndarray)) and len(qrs) == 100:
                qrs = np.asarray(qrs, dtype=float)
                if len(past_qrs) >= min_history:
                    ref = np.median(np.vstack(past_qrs), axis=0)
                    df.at[idx[i], "QRS_Global"] = float(np.mean((qrs - ref) ** 2))
                past_qrs.append(qrs)

            # ---- z-scoring ----
            if i < min_history:
                continue

            past_idx = idx[:i]

            for field in z_fields + ["TMV_Global", "QRS_Global"]:
                cur = df.at[idx[i], field]
                if pd.isna(cur):
                    continue

                past_vals = df.loc[past_idx, field].dropna()
                if len(past_vals) < min_history:
                    continue

                q25, q75 = np.percentile(past_vals, [25, 75])
                iqr = q75 - q25
                if iqr < 1e-6:
                    continue

                med = np.median(past_vals)
                z = (cur - med) / (iqr / 1.349 + 1e-6)
                df.at[idx[i], f"{field}_Z"] = np.clip(z, -10, 10)

            # ---- availability (FIX HERE) ----
            row_z = df.loc[idx[i], z_cols].astype(float)  # ← THIS LINE FIXES IT
            nan_frac = row_z.isna().mean()

            df.loc[idx[i], z_cols] = row_z.fillna(0)
            df.at[idx[i], "avail"] = int(nan_frac <= nan_frac_threshold)

    return df


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


def plot_standardized_qt_tmv_with_single_regression(
    results_df,
    alarms_df,
    features,
    rf_model_path_reg="random_forest_vtac_model_regression.joblib",
    z_threshold=1.5,
    output_dir="plots",
    smooth_kernel=3,
):

    os.makedirs(output_dir, exist_ok=True)

    # Load SINGLE regression model
    rf_model_reg = joblib.load(rf_model_path_reg)

    unique_records = sorted(results_df["Record"].unique())

    for record_id in unique_records:
        print(f"\n[INFO] Processing record: {record_id}")

        record_df = results_df[results_df["Record"] == record_id].reset_index(drop=True)

        missing_features = [f for f in features if f not in record_df.columns]
        if record_df.empty or missing_features:
            print(f"[SKIP] {record_id}: Missing features or empty DF.")
            continue

        record_df["Start"] = pd.to_datetime(record_df["Start"], errors="coerce")
        if record_df["Start"].isna().all():
            print(f"[SKIP] {record_id}: Invalid Start times.")
            continue

        # --------------------
        # Regression prediction
        # --------------------
        X = record_df[features]
        record_df["VTAC_Risk"] = rf_model_reg.predict(X)

        # Temporal smoothing
        if smooth_kernel and smooth_kernel > 1:
            record_df["VTAC_Risk"] = medfilt(
                record_df["VTAC_Risk"].values,
                kernel_size=smooth_kernel
            )

        # Drop invalid rows
        valid_mask = X.notna().all(axis=1) & record_df["VTAC_Risk"].notna()
        record_df = record_df[valid_mask].reset_index(drop=True)

        if record_df.empty:
            print(f"[SKIP] {record_id}: No valid windows.")
            continue

        # Time axis
        window_times = (
            record_df["Start"] - record_df["Start"].iloc[0]
        ).dt.total_seconds()

        record_start = record_df["Start"].iloc[0]
        record_alarms = alarms_df[alarms_df["Files"] == record_id]

        # --------------------
        # Z-score fields
        # --------------------
        z_fields = [
            ("TMV_Score_Z", "darkblue", "TMV Score (Z)"),
            ("QT_Interval_Z", "purple", "QT Interval (Z)"),
            ("Mean_HR_Z", "green", "Mean HR (Z)"),
            ("TMV_Global_Z", "crimson", "TMV Global (Z)"),
            ("QRS_Duration_Z", "slateblue", "QRS Duration (Z)"),
        ]

        total_plots = 1 + len(z_fields) + 1  # raw + Z + regression

        fig, axes = plt.subplots(
            total_plots, 1,
            figsize=(40, 2.2 * total_plots),
            sharex=True
        )

        # --------------------
        # RAW ECG
        # --------------------
        for i, row in record_df.iterrows():
            ecg_raw = row.get("ECG_Raw", None)
            if isinstance(ecg_raw, list) and len(ecg_raw) > 0:
                t = np.linspace(window_times[i], window_times[i] + 30, len(ecg_raw))
                axes[0].plot(t, ecg_raw, alpha=0.4)

        axes[0].set_ylabel("Raw ECG")
        axes[0].set_ylim(-500, 500)
        axes[0].set_title(f"Record: {record_id}")

        # --------------------
        # Helper: shade Z excursions
        # --------------------
        def shade_z(ax, vals, times):
            for i, z in enumerate(vals):
                if pd.isna(z):
                    continue
                if z > z_threshold:
                    ax.axvspan(times[i], times[i] + 30, color="red", alpha=0.2)
                elif z < -z_threshold:
                    ax.axvspan(times[i], times[i] + 30, color="blue", alpha=0.2)

        # --------------------
        # Z-score panels
        # --------------------
        for j, (field, color, label) in enumerate(z_fields):
            ax = axes[j + 1]
            ax.plot(window_times, record_df[field], color=color)
            ax.axhline(z_threshold, color="red", linestyle="--")
            ax.axhline(-z_threshold, color="blue", linestyle="--")
            shade_z(ax, record_df[field], window_times)
            ax.set_ylabel(label)

        # --------------------
        # VTAC regression risk
        # --------------------
        reg_ax = axes[-1]
        reg_ax.plot(window_times, record_df["VTAC_Risk"], color="orange", label="VTAC Risk")
        reg_ax.set_ylim(0, 1)
        reg_ax.set_ylabel("Risk")
        reg_ax.set_xlabel("Time (s)")
        reg_ax.set_title("Predicted VTAC Risk (RF Regression)")
        reg_ax.legend()

        # --------------------
        # VTAC markers
        # --------------------
        max_time = window_times.max() + 30

        for ax in axes:
            ax.set_xlim(0, max_time)
            for _, alarm in record_alarms.iterrows():
                vt_start = pd.to_datetime(alarm["StartTime"], errors="coerce")
                if pd.isna(vt_start):
                    continue
                vt_end = vt_start + timedelta(seconds=int(alarm["Duration"]))
                s = (vt_start - record_start).total_seconds()
                e = (vt_end - record_start).total_seconds()
                ax.axvline(s, color="black", linestyle="--")
                ax.axvline(e, color="black", linestyle="--")

        plt.tight_layout()
        plt.show()

def plot_subject_panels(
    processed_df,
    sampling_rate_default=250,
    window_len_sec=30,
    z_threshold=1.5,
):
    """
    Plot Raw ECG + Z-scored features from compute_refs_and_zscores() output.

    Rules:
    - DO NOT fill warm-up NaNs
    - CU*    : warmup = 12 windows
    - Case_* : warmup = 60 windows
    - FID*   : warmup = 60 windows
    - After warm-up:
        • If avail == 0 → Z-values should already be 0
        • If avail == 1 → real Z-values (may exceed threshold)
    """


    # -------------------------------
    # Z-scored feature list
    # -------------------------------
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

    records = sorted(processed_df["Record"].unique())

    for record_id in records:

        df = (
            processed_df[processed_df["Record"] == record_id]
            .sort_values("Start")
            .reset_index(drop=True)
        )

        if df.empty:
            continue

        # -------------------------------
        # Sampling rate
        # -------------------------------
        sr = sampling_rate_default if record_id.lower().startswith("cu") else 240

        # -------------------------------
        # Warm-up logic
        # -------------------------------
        rid = str(record_id)
        if rid.lower().startswith("cu"):
            warmup_n = 12
        elif rid.startswith("Case_") or rid.startswith("FID"):
            warmup_n = 60
        else:
            warmup_n = 12

        time_sec = df["Start"] / sr

        # -------------------------------
        # Create figure
        # -------------------------------
        n_panels = len(z_fields) + 1
        fig, axes = plt.subplots(
            n_panels, 1,
            figsize=(14, 2.8 * n_panels),
            sharex=True
        )

        # =====================================================
        # Panel 0 — Raw ECG
        # =====================================================
        for i, row in df.iterrows():
            ecg = row.get("ECG_Raw")
            if isinstance(ecg, list) and len(ecg) > 0:
                t = np.linspace(
                    time_sec.iloc[i],
                    time_sec.iloc[i] + window_len_sec,
                    len(ecg)
                )
                axes[0].plot(t, ecg, color="black", alpha=0.35)

        axes[0].set_ylabel("ECG")
        axes[0].set_title(f"Record: {record_id}")

        # =====================================================
        # Helper: Z-plot
        # =====================================================
        def plot_z(ax, z_series, label):
            s = z_series.copy()

            # --- Enforce warm-up NaNs visually ---
            s.iloc[:warmup_n] = np.nan

            ax.plot(time_sec, s, color="tab:blue", linewidth=1)

            # --- Highlight threshold crossings ---
            for i in range(len(s)):
                if pd.notna(s.iloc[i]) and s.iloc[i] > z_threshold:
                    ax.axvspan(
                        time_sec.iloc[i],
                        time_sec.iloc[i] + window_len_sec,
                        color="red",
                        alpha=0.2
                    )

            ax.axhline(z_threshold, linestyle="--", color="gray", linewidth=1)
            ax.set_ylabel(label)

        # =====================================================
        # Plot Z-features
        # =====================================================
        for k, field in enumerate(z_fields, start=1):
            z_col = f"{field}_Z"
            if z_col in df.columns:
                plot_z(axes[k], df[z_col], z_col)

        # =====================================================
        # VTAC markers
        # =====================================================
        vtac_rows = df[df["VTAC_Label"] == 1]
        if not vtac_rows.empty:
            vtac_start = vtac_rows["Start"].min() / sr
            vtac_end   = vtac_rows["Start"].max() / sr

            for ax in axes:
                ax.axvline(vtac_start, color="black", linestyle="--", alpha=0.8)
                ax.axvline(vtac_end,   color="black", linestyle="--", alpha=0.8)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()
