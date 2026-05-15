import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from datetime import timedelta
from scipy.signal import medfilt

_PLOT_Z_FIELDS = [
    ("QT_Interval_Z", "purple",    "QT Interval (Z)"),
    ("Mean_HR_Z",     "green",     "Mean HR (Z)"),
    ("TMV_Global_Z",  "crimson",   "TMV Global (Z)"),
    ("QRS_Duration_Z","slateblue", "QRS Duration (Z)"),
    ("Clip_Ratio",    "black",     "Clip Ratio"),
]

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
            row_z = df.loc[idx[i], z_cols].astype(float)
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
    smooth_kernel=3,
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

        record_df = results_df[results_df["Record"] == record_id].reset_index(drop=True)

        missing_features = [f for f in features if f not in record_df.columns]
        if record_df.empty or missing_features:
            print(f"[SKIP] {record_id}: Missing features or empty DF.")
            continue

        record_df["Start"] = pd.to_datetime(record_df["Start"], errors="coerce")
        if record_df["Start"].isna().all():
            print(f"[SKIP] {record_id}: All Start times are NaT.")
            continue

        X = record_df[features]
        record_df["Prob_RF"] = rf_model.predict_proba(X)[:, 1]
        record_df["Prob_XGB"] = xgb_model.predict_proba(X)[:, 1]
        record_df["Reg_RF"] = rf_model_reg.predict(X)
        record_df["Reg_XGB"] = xgb_model_reg.predict(X)

        if smooth_kernel and smooth_kernel > 1:
            record_df["Prob_RF"] = medfilt(record_df["Prob_RF"].values, kernel_size=smooth_kernel)
            record_df["Reg_RF"]  = medfilt(record_df["Reg_RF"].values,  kernel_size=smooth_kernel)
            record_df["Prob_XGB"] = medfilt(record_df["Prob_XGB"].values, kernel_size=smooth_kernel)
            record_df["Reg_XGB"]  = medfilt(record_df["Reg_XGB"].values,  kernel_size=smooth_kernel)

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

        window_times = (record_df["Start"] - record_df["Start"].iloc[0]).dt.total_seconds()
        record_alarms = alarms_df[alarms_df["Files"] == record_id]
        record_start = record_df["Start"].iloc[0]

        z_fields = _PLOT_Z_FIELDS
        num_z = len(z_fields)
        total_plots = 1 + num_z + 2

        fig, axes = plt.subplots(
            total_plots, 1,
            figsize=(40, 2.2 * total_plots),
            sharex=True
        )

        # ----- RAW ECG -----
        for i, row in record_df.iterrows():
            ecg_raw = row.get("ECG_Raw", None)
            if isinstance(ecg_raw, list) and len(ecg_raw) > 0:
                t = np.linspace(window_times[i], window_times[i] + 30, len(ecg_raw))
                axes[0].plot(t, ecg_raw, alpha=0.5)

        axes[0].set_ylabel("Raw ECG")
        axes[0].set_ylim(-500, 500)
        axes[0].set_title(f"Record: {record_id}")

        for j, (field, color, label) in enumerate(z_fields):
            ax = axes[j + 1]
            ax.plot(window_times, record_df[field], color=color)
            ax.set_ylabel(label)

            if field != "Clip_Ratio":
                ax.axhline(z_threshold, color="red", linestyle="--")
                ax.axhline(-z_threshold, color="blue", linestyle="--")
            else:
                ax.set_ylim(0, 0.5)
                ax.axhline(0.15, color="red", linestyle="--", alpha=0.7)

        prob_ax = axes[1 + num_z]
        prob_ax.plot(window_times, record_df["Prob_RF"], color="orange", label="RF Prob")
        prob_ax.plot(window_times, record_df["Prob_XGB"], color="red", linestyle="--", label="XGB Prob")
        prob_ax.set_ylim(0, 1)
        prob_ax.set_title("Predicted VTAC Probability")
        prob_ax.legend()

        reg_ax = axes[2 + num_z]
        reg_ax.plot(window_times, record_df["Reg_RF"], color="orange", label="RF Reg")
        reg_ax.plot(window_times, record_df["Reg_XGB"], color="red", linestyle="--", label="XGB Reg")
        reg_ax.set_ylim(0, 1)
        reg_ax.set_title("Predicted VTAC Risk")
        reg_ax.set_xlabel("Time (s)")
        reg_ax.legend()

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

        # ==========================
        # SAVE FIGURE  ✅
        # ==========================
        save_path = os.path.join(
            output_dir,
            f"{record_id}_dual_prediction_plot.png"
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
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

        if smooth_kernel and smooth_kernel > 1:
            record_df["VTAC_Risk"] = medfilt(
                record_df["VTAC_Risk"].values,
                kernel_size=smooth_kernel
            )

        valid_mask = X.notna().all(axis=1) & record_df["VTAC_Risk"].notna()
        record_df = record_df[valid_mask].reset_index(drop=True)

        if record_df.empty:
            print(f"[SKIP] {record_id}: No valid windows.")
            continue

        window_times = (
            record_df["Start"] - record_df["Start"].iloc[0]
        ).dt.total_seconds()

        record_start = record_df["Start"].iloc[0]
        record_alarms = alarms_df[alarms_df["Files"] == record_id]

        z_fields = _PLOT_Z_FIELDS
        total_plots = 1 + len(z_fields) + 1

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

        for j, (field, color, label) in enumerate(z_fields):
            ax = axes[j + 1]
            ax.plot(window_times, record_df[field], color=color)
            ax.set_ylabel(label)

            if field != "Clip_Ratio":
                ax.axhline(z_threshold, color="red", linestyle="--")
                ax.axhline(-z_threshold, color="blue", linestyle="--")
            else:
                ax.set_ylim(0, 0.5)
                ax.axhline(0.15, color="red", linestyle="--", alpha=0.7)


        reg_ax = axes[-1]
        reg_ax.plot(window_times, record_df["VTAC_Risk"], color="orange", label="VTAC Risk")
        reg_ax.set_ylim(0, 1)
        reg_ax.set_ylabel("Risk")
        reg_ax.set_xlabel("Time (s)")
        reg_ax.set_title("Predicted VTAC Risk (RF Regression)")
        reg_ax.legend()

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

        # ==========================
        # SAVE FIGURE  ✅
        # ==========================
        out_path = os.path.join(
            output_dir,
            f"{record_id}_standardized_qt_tmv_regression.png"
        )

        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"[SAVED] {out_path}")


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
