import os
import h5py
import numpy as np
import pandas as pd
import neurokit2 as nk

from tqdm import tqdm
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import skew
from datetime import datetime, timedelta


# ==============================
# T-wave feature extraction
# ==============================
def extract_t_wave_features(t_wave_segment, sampling_rate=250):
    time = np.arange(len(t_wave_segment)) / sampling_rate
    peak_idx = np.argmax(t_wave_segment)

    t1 = peak_idx
    t2 = len(t_wave_segment) - peak_idx
    asymmetry = abs(t1 - t2) / (t1 + t2 + 1e-6)

    peaks, _ = find_peaks(t_wave_segment, distance=10, prominence=0.05)
    notch = 1 if len(peaks) > 1 else 0

    slopes = np.abs(np.diff(t_wave_segment))
    flatness = 1 / (np.percentile(slopes, 95) + 1e-6)

    mcs = asymmetry + notch + 1.6 * flatness

    return {
        "asymmetry": asymmetry,
        "notch": notch,
        "flatness": flatness,
        "mcs": mcs,
    }


# ==============================
# Bandpass filter
# ==============================
def bandpass_filter(signal, lowcut=0.2, highcut=30, fs=250, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def autocorr_metrics(
    x, fs, search_min_bpm=50, search_max_bpm=200, mean_window_frac=0.1
):
    """
    Raw-signal autocorr metrics in a HR-constrained lag band.
    Returns: (peak_value, lag_seconds, mean_around_peak)
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 8 or not np.isfinite(x).all() or np.allclose(np.std(x), 0):
        return np.nan, np.nan, np.nan

    # normalize
    x = (x - np.mean(x)) / (np.std(x) + 1e-9)
    acf = np.correlate(x, x, mode="full")
    acf = acf[len(acf) // 2 :]
    if acf[0] == 0:
        return np.nan, np.nan, np.nan
    acf = acf / (acf[0] + 1e-9)

    # search lag range
    min_lag = int(np.floor((60.0 / max(search_max_bpm, 1e-3)) * fs))
    max_lag = int(np.ceil((60.0 / max(search_min_bpm, 1e-3)) * fs))
    min_lag = max(min_lag, 1)
    max_lag = min(max_lag, len(acf) - 1)
    if max_lag <= min_lag:
        return np.nan, np.nan, np.nan

    seg = acf[min_lag : max_lag + 1]
    k_rel = int(np.argmax(seg))
    k = min_lag + k_rel
    peak = float(acf[k])

    w = max(int((max_lag - min_lag + 1) * mean_window_frac), 1)
    s = max(k - w, 1)
    e = min(k + w + 1, len(acf))
    mean_win = float(np.mean(acf[s:e]))
    lag_sec = float(k / fs)

    return peak, lag_sec, mean_win


def rr_autocorr_metrics(r_peaks, fs, mean_window_beats=1):
    """
    RR-series autocorr metrics (rhythm regularity).
    Returns: (peak_value, lag_in_beats, mean_around_peak)
    """
    r_peaks = np.asarray(r_peaks, dtype=int)
    if len(r_peaks) < 4:
        return np.nan, np.nan, np.nan

    rr = np.diff(r_peaks) / float(fs)  # seconds
    if len(rr) < 3 or not np.isfinite(rr).all() or np.allclose(np.std(rr), 0):
        return np.nan, np.nan, np.nan

    rr = rr - np.mean(rr)
    acf = np.correlate(rr, rr, mode="full")
    acf = acf[len(rr) - 1 :]
    if acf[0] == 0:
        return np.nan, np.nan, np.nan
    acf = acf / (acf[0] + 1e-9)

    # search 1–10 beats lag
    min_lag = 1
    max_lag = min(10, len(acf) - 1)
    if max_lag <= min_lag:
        return np.nan, np.nan, np.nan

    seg = acf[min_lag : max_lag + 1]
    k_rel = int(np.argmax(seg))
    k = min_lag + k_rel
    peak = float(acf[k])

    w = max(int(mean_window_beats), 1)
    s = max(k - w, 1)
    e = min(k + w + 1, len(acf))
    mean_win = float(np.mean(acf[s:e]))

    return peak, float(k), mean_win


# ==============================
# TMV and QT calculation (now returns QRS_Wave resampled to 100)
# ==============================
def calculate_tmv_and_qt(
    ecg_segment, sampling_rate=250, j_offset_ms=60, baseline_ms=80, slope_window_ms=40
):
    """
    Returns (22 items):
      1.  avg_qt
      2.  subject_twave
      3.  ecg_segment
      4.  mean_hr
      5.  rmssd
      6.  sdnn
      7.  t_wave_features (dict)
      8.  tmv_score
      9.  max_hr
      10. min_hr
      11. qrs_duration
      12. qrs_area
      13. qrs_skewness
      14. st_deviation_mean
      15. st_slope_mean
      16. subject_qrs (QRS_Wave, len 100)
      17. ac_peak               <-- NEW (raw ECG ACF peak within HR band)
      18. ac_lag_sec            <-- NEW (lag of that peak in seconds)
      19. ac_mean               <-- NEW (mean ACF around peak)
      20. rr_ac_peak            <-- NEW (RR ACF peak)
      21. rr_ac_lag_beats       <-- NEW (lag in beats)
      22. rr_ac_mean            <-- NEW (mean ACF around RR peak)
    """
    try:
        ecg_segment = np.array(ecg_segment)
        if np.isnan(ecg_segment).any() or np.std(ecg_segment) == 0:
            return (
                np.nan,
                [],
                ecg_segment.tolist(),
                np.nan,
                np.nan,
                np.nan,
                {},
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                [],
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )

        ecg_filtered = bandpass_filter(
            ecg_segment, lowcut=0.2, highcut=30, fs=sampling_rate
        )
        ecg_cleaned = nk.ecg_clean(ecg_filtered, sampling_rate=sampling_rate)
        _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        r_peaks = info.get("ECG_R_Peaks", [])
        if len(r_peaks) < 3:
            return (
                np.nan,
                [],
                ecg_segment.tolist(),
                np.nan,
                np.nan,
                np.nan,
                {},
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                [],
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )

        # Delineation
        _, delineate = nk.ecg_delineate(
            ecg_cleaned, r_peaks, sampling_rate=sampling_rate, method="peak"
        )
        q_peaks = delineate.get("ECG_Q_Peaks", [])
        t_peaks = delineate.get("ECG_T_Peaks", [])
        s_peaks = delineate.get("ECG_S_Peaks", [])

        qt_intervals, t_waves = [], []
        qrs_durations, qrs_areas, qrs_skewness_list = [], [], []
        qrs_waves = []
        st_deviations, st_slopes = [], []
        j_offset = int((j_offset_ms / 1000.0) * sampling_rate)
        baseline_n = int((baseline_ms / 1000.0) * sampling_rate)
        slope_n = int((slope_window_ms / 1000.0) * sampling_rate)

        for q, t, s in zip(q_peaks, t_peaks, s_peaks):
            if q is None or t is None or s is None:
                continue
            try:
                q, t, s = int(q), int(t), int(s)
            except (ValueError, TypeError):
                continue
            if (
                np.isnan(q)
                or np.isnan(t)
                or t <= q
                or (t - q) < 20
                or t >= len(ecg_cleaned)
            ):
                continue

            # QT
            qt_intervals.append((t - q) / sampling_rate)

            # T-wave (S->T), resample 100, z-norm
            segment_t = ecg_cleaned[s:t]
            if len(segment_t) >= 8:
                t_resampled = np.interp(
                    np.linspace(0, len(segment_t) - 1, 100),
                    np.arange(len(segment_t)),
                    segment_t,
                )
                t_norm = (t_resampled - np.mean(t_resampled)) / (
                    np.std(t_resampled) + 1e-6
                )
                t_waves.append(t_norm)

            # QRS features + QRS_Wave (Q->S)
            if 0 <= q < s < len(ecg_cleaned) and (s - q) > 5:
                qrs_durations.append((s - q) / sampling_rate)
                qrs_areas.append(
                    np.trapz(np.abs(ecg_cleaned[q:s]), dx=1 / sampling_rate)
                )
                qrs_wave_raw = ecg_cleaned[q:s]
                if len(qrs_wave_raw) >= 6 and not np.isnan(qrs_wave_raw).any():
                    qrs_resampled = np.interp(
                        np.linspace(0, len(qrs_wave_raw) - 1, 100),
                        np.arange(len(qrs_wave_raw)),
                        qrs_wave_raw,
                    )
                    qrs_norm = (qrs_resampled - np.mean(qrs_resampled)) / (
                        np.std(qrs_resampled) + 1e-6
                    )
                    qrs_waves.append(qrs_norm)
                    qrs_skewness_list.append(skew(qrs_wave_raw))

            # ST deviation & slope
            b_start = max(q - baseline_n, 0)
            b_end = max(q, 0)
            baseline_val = (
                np.median(ecg_cleaned[b_start:b_end])
                if (b_end - b_start) >= 5
                else np.nan
            )
            j_idx = min(s + j_offset, len(ecg_cleaned) - 1)
            st_val = (
                (ecg_cleaned[j_idx] - baseline_val)
                if not np.isnan(baseline_val)
                else np.nan
            )
            st_deviations.append(st_val)
            j2 = min(j_idx + slope_n, len(ecg_cleaned) - 1)
            st_slope = (
                (ecg_cleaned[j2] - ecg_cleaned[j_idx]) / ((j2 - j_idx) / sampling_rate)
                if j2 > j_idx
                else np.nan
            )
            st_slopes.append(st_slope)

        if len(t_waves) < 3:
            return (
                np.nan,
                [],
                ecg_segment.tolist(),
                np.nan,
                np.nan,
                np.nan,
                {},
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                [],
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )

        avg_qt = np.mean(qt_intervals)
        subject_twave = np.mean(t_waves, axis=0)

        # HR / HRV
        hr = nk.ecg_rate(r_peaks, sampling_rate=sampling_rate)
        mean_hr = np.mean(hr) if len(hr) > 0 else np.nan
        max_hr = np.max(hr) if len(hr) > 0 else np.nan
        min_hr = np.min(hr) if len(hr) > 0 else np.nan
        rr_intervals = np.diff(r_peaks) / sampling_rate * 1000.0
        rmssd = (
            np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
            if len(rr_intervals) > 1
            else np.nan
        )
        sdnn = np.std(rr_intervals) if len(rr_intervals) > 1 else np.nan

        # T-wave features & TMV
        t_wave_features = extract_t_wave_features(subject_twave, sampling_rate)
        diffs = [(tw - subject_twave) ** 2 for tw in t_waves]
        mse_vals = [np.mean(d) for d in diffs if not np.any(np.isnan(d))]
        tmv_score = np.mean(mse_vals) if mse_vals else np.nan

        # T-wave amplitude variability
        t_wave_amplitudes = [
            np.max(tw) - np.min(tw) for tw in t_waves if not np.any(np.isnan(tw))
        ]
        if len(t_wave_amplitudes) >= 3:
            t_wave_amp_std = np.std(t_wave_amplitudes)
            t_wave_amp_cv = t_wave_amp_std / (np.mean(t_wave_amplitudes) + 1e-6)
        else:
            t_wave_amp_std, t_wave_amp_cv = np.nan, np.nan
        t_wave_features["TWAmp_Std"] = t_wave_amp_std
        t_wave_features["TWAmp_CV"] = t_wave_amp_cv

        # QRS aggregates
        qrs_duration = np.mean(qrs_durations) if qrs_durations else np.nan
        qrs_area = np.mean(qrs_areas) if qrs_areas else np.nan
        qrs_skewness = np.mean(qrs_skewness_list) if qrs_skewness_list else np.nan

        # ST aggregates
        st_deviation_mean = np.nanmean(st_deviations) if len(st_deviations) else np.nan
        st_slope_mean = np.nanmean(st_slopes) if len(st_slopes) else np.nan

        # Averaged QRS_Wave (len 100)
        if len(qrs_waves) >= 3:
            subject_qrs = np.mean(qrs_waves, axis=0).tolist()
        elif len(qrs_waves) > 0:
            subject_qrs = np.mean(qrs_waves, axis=0).tolist()
        else:
            subject_qrs = []

        # --------- NEW: Autocorrelations ---------
        # Raw ECG (morphology periodicity) ACF in HR band
        ac_peak, ac_lag_sec, ac_mean = autocorr_metrics(ecg_cleaned, sampling_rate)
        # RR interval series (rhythm regularity) ACF
        rr_ac_peak, rr_ac_lag_beats, rr_ac_mean = rr_autocorr_metrics(
            r_peaks, sampling_rate
        )

        return (
            avg_qt,
            subject_twave.tolist(),
            ecg_segment.tolist(),
            mean_hr,
            rmssd,
            sdnn,
            t_wave_features,
            tmv_score,
            max_hr,
            min_hr,
            qrs_duration,
            qrs_area,
            qrs_skewness,
            st_deviation_mean,
            st_slope_mean,
            subject_qrs,
            ac_peak,
            ac_lag_sec,
            ac_mean,
            rr_ac_peak,
            rr_ac_lag_beats,
            rr_ac_mean,
        )

    except Exception as e:
        print(f"[ERROR] {e}")
        return (
            np.nan,
            [],
            ecg_segment.tolist(),
            np.nan,
            np.nan,
            np.nan,
            {},
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            [],
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )


# ==============================
# Extend last VTAC per Record (inclusive overlap; units: samples)
# ==============================
def extend_last_vtac_label_inplace(df, sampling_rate=250, extension_sec=30):
    """
    Extend the last VTAC segment per Record by `extension_sec` seconds.
    Assumes Start/End are in SAMPLES.
    Marks any window that overlaps the extended VTAC period.
    """
    df = df.copy()
    ext_samples = int(extension_sec * sampling_rate)
    label_upper = df["Label"].astype(str).str.upper()

    for rec, g in df.groupby("Record"):
        g_vtac = g[label_upper.loc[g.index] == "VTAC"].sort_values("Start")
        if g_vtac.empty:
            continue

        last_start = g_vtac["Start"].iloc[-1]
        last_end = g_vtac["End"].iloc[-1]
        extended_end = last_end + ext_samples

        # Overlap-based relabel
        mask = (
            (df["Record"] == rec)
            & (df["End"] >= last_start)
            & (df["Start"] <= extended_end)
        )
        df.loc[mask, "Label"] = "VTAC"

    return df


# ==============================
# Main processing function (simple signature, returns DataFrame)
# ==============================
def process_dataframe(df, sampling_rate=250, extension_sec=30):
    """
    1) Extend the last VTAC per record by `extension_sec` seconds (in samples).
    2) Print VTAC ratio change per subject.
    3) Extract features per window (now includes QRS_Wave of length 100).
    4) Return a single pandas DataFrame.
    """
    # VTAC ratio before extension (per Record)
    before_ratio = df.groupby("Record")["Label"].apply(
        lambda x: (x.astype(str).str.upper() == "VTAC").mean() * 100
    )

    # Apply VTAC extension
    df_ext = extend_last_vtac_label_inplace(
        df, sampling_rate=sampling_rate, extension_sec=extension_sec
    )

    # VTAC ratio after extension (per Record)
    after_ratio = df_ext.groupby("Record")["Label"].apply(
        lambda x: (x.astype(str).str.upper() == "VTAC").mean() * 100
    )

    # Delta
    delta_ratio = after_ratio - before_ratio

    # Combine into DataFrame and print
    ratio_df = pd.DataFrame(
        {
            "VTAC_Ratio_Before(%)": before_ratio,
            "VTAC_Ratio_After(%)": after_ratio,
            "Delta(After-Before)": delta_ratio,
        }
    )
    print("\n=== VTAC Ratio Change Per Subject ===")
    print(ratio_df.to_string())

    results = []
    for _, row in tqdm(df_ext.iterrows(), total=len(df_ext)):
        (
            qt,
            t_wave,
            raw_ecg,
            mean_hr,
            rmssd,
            sdnn,
            t_wave_features,
            tmv_score,
            max_hr,
            min_hr,
            qrs_dur,
            qrs_area,
            qrs_skew,
            st_dev_mean,
            st_slope_mean,
            qrs_wave,
            ac_ecg_peak,
            ac_ecg_lag_sec,
            ac_ecg_mean,
            ac_rr_peak,
            ac_rr_lag_beats,
            ac_rr_mean,
        ) = calculate_tmv_and_qt(row["ECG"], sampling_rate)

        results.append(
            {
                "Record": row["Record"],
                "Start": int(row["Start"]),
                "End": int(row["End"]),
                "Label": row["Label"],
                "QT_Interval": qt,
                "T_Wave": t_wave,
                "QRS_Wave": qrs_wave,
                "ECG_Raw": raw_ecg,
                "TMV_Score": tmv_score,
                "Mean_HR": mean_hr,
                "RMSSD": rmssd,
                "SDNN": sdnn,
                "Max_HR": max_hr,
                "Min_HR": min_hr,
                "T_Flatness": t_wave_features.get("flatness"),
                "T_Asymmetry": t_wave_features.get("asymmetry"),
                "T_Notch": t_wave_features.get("notch"),
                "T_MCS": t_wave_features.get("mcs"),
                "TWAmp_Std": t_wave_features.get("TWAmp_Std"),
                "TWAmp_CV": t_wave_features.get("TWAmp_CV"),
                "QRS_Duration": qrs_dur,
                "QRS_Area": qrs_area,
                "QRS_Skewness": qrs_skew,
                "ST_Deviation_Mean": st_dev_mean,
                "ST_Slope_Mean": st_slope_mean,
                "AC_ECG_Peak": ac_ecg_peak,
                "AC_ECG_Lag_Sec": ac_ecg_lag_sec,
                "AC_ECG_MeanAroundPeak": ac_ecg_mean,
                "AC_RR_Peak": ac_rr_peak,
                "AC_RR_Lag_Beats": ac_rr_lag_beats,
                "AC_RR_MeanAroundPeak": ac_rr_mean,
            }
        )

    return pd.DataFrame(results)


def create_windowed_ecg_from_mat(
    alarms_df,
    record_file,
    waveform_dir="VTSampleData/waveform",
    sampling_rate=240,
    window_duration=30,
    window_shift=5,
    pre_buffer_sec=3600,
    post_buffer_sec=500,
):
    """
    Create 30s / 5s-shift sliding ECG windows from MAT waveform files,
    but ONLY around the FIRST VTAC EVENT:

        • includes 3600 seconds BEFORE the first VTAC
        • includes 500 seconds AFTER the first VTAC
        • ignores ALL other VTAC events

    Returns:
        pandas.DataFrame with columns:
            ['Record', 'Start', 'End', 'Label', 'ECG']
    """

    wf_path = f"{waveform_dir}/{record_file}.mat"
    if not os.path.exists(wf_path):
        print(f"[SKIP] {wf_path} not found.")
        return None

    # -----------------------------------------------------
    # Load waveform from .mat
    # -----------------------------------------------------
    with h5py.File(wf_path, "r") as f_wave:
        ecg = np.array(f_wave["ECG2w"]).squeeze()
        ecg_time = np.array(f_wave["ECG2w_time"]).squeeze()

    # Convert MATLAB serial time -> Python datetime
    segment_start_times = np.array(
        [
            datetime.fromordinal(int(dn)) + timedelta(days=dn % 1) - timedelta(days=366)
            for dn in ecg_time
        ]
    )
    samples_per_segment = ecg.shape[1]

    # Build full ECG timeline
    full_time_vector = []
    for start_time in segment_start_times:
        full_time_vector.extend(
            [
                start_time + timedelta(seconds=i / sampling_rate)
                for i in range(samples_per_segment)
            ]
        )
    full_ecg = ecg.flatten()

    # -----------------------------------------------------
    # Find ALL VTAC events for this record
    # -----------------------------------------------------
    alarms_record = alarms_df[alarms_df["Files"] == record_file]

    vt_events = []
    for _, alarm in alarms_record.iterrows():
        vt_start = pd.to_datetime(
            alarm["StartTime"], format="%m/%d/%y %H:%M", errors="coerce"
        )
        vt_end = vt_start + timedelta(seconds=int(alarm["Duration"]))
        vt_events.append((vt_start, vt_end))

    if not vt_events:
        return None

    # -----------------------------------------------------
    # Keep ONLY the FIRST VTAC EVENT
    # -----------------------------------------------------
    first_vt_start, first_vt_end = sorted(vt_events, key=lambda x: x[0])[0]

    vtach_intervals = [(first_vt_start, first_vt_end)]

    # -----------------------------------------------------
    # Restrict extraction region to:
    #   FIRST_VTAC_START - 3600 sec
    #   FIRST_VTAC_END + 500 sec
    # -----------------------------------------------------
    plot_start = first_vt_start - timedelta(seconds=pre_buffer_sec)
    plot_end = first_vt_end + timedelta(seconds=post_buffer_sec)

    # Mask data inside region
    mask = [(t >= plot_start and t <= plot_end) for t in full_time_vector]
    plot_times = np.array(full_time_vector)[mask]
    plot_ecg = full_ecg[mask]

    # -----------------------------------------------------
    # Sliding Windowing
    # -----------------------------------------------------
    window_size = window_duration * sampling_rate     # 30 sec × 240 Hz = 7200 samples
    step_size = window_shift * sampling_rate          # 5 sec × 240 Hz = 1200 samples

    windowed_data = []

    for start_idx in range(0, len(plot_ecg) - window_size + 1, step_size):
        segment = plot_ecg[start_idx : start_idx + window_size]
        start_time = plot_times[start_idx]
        end_time = plot_times[start_idx + window_size - 1]

        # Label each window: VTAC or Normal
        label = "Normal"
        for vt_start, vt_end in vtach_intervals:
            if vt_start <= start_time <= vt_end:
                label = "VTAC"
                break

        windowed_data.append(
            {
                "Record": record_file,
                "Start": start_time,
                "End": end_time,
                "Label": label,
                "ECG": segment,
            }
        )

    return pd.DataFrame(windowed_data)

def convert_and_relabel_windowed_df_full(
    df_full,
    vtac_intervals,
    sampling_rate=240,
    win_sec=30,
    shift_sec=5
):
    """
    Convert timestamp-based windowed_df_full into sample-based windowed_df,
    and relabel using VTAC / Pre-VTAC / Other logic.
    """

    W = win_sec * sampling_rate     # 7200 samples
    S = shift_sec * sampling_rate   # 1200 samples

    df = df_full.copy().reset_index(drop=True)

    # ------------------------------------------------------
    # 1) Convert timestamps -> sample Start/End based on index
    # ------------------------------------------------------
    df["Start"] = df.index * S
    df["End"] = df["Start"] + W

    # Keep only needed columns
    df = df[["Record", "Start", "End", "Label", "ECG"]]

    # ------------------------------------------------------
    # 2) Relabel to VTAC / Pre-VTAC / Other
    # ------------------------------------------------------
    new_labels = []

    for _, row in df.iterrows():
        start = row["Start"]
        end = row["End"]

        # Here we convert sample-based Start to time for comparison
        # But better: convert VTAC intervals to sample indices too
        # Let’s assume df_full had timestamp Start
        # We fetch the original timestamp row using df_full
        original_time = df_full.loc[row.name, "Start"]

        # --- VTAC: if original_time is inside any VTAC interval ---
        in_vtac = any(vs <= original_time <= ve for vs, ve in vtac_intervals)
        if in_vtac:
            new_labels.append("VTAC")
            continue

        # --- Pre-VTAC ---
        next_vt_starts = [vs for vs, _ in vtac_intervals if vs > original_time]
        if next_vt_starts:
            next_vt = min(next_vt_starts)
            if original_time < next_vt:
                new_labels.append("Pre-VTAC")
                continue

        # Default: Other
        new_labels.append("Other")

    df["Label"] = new_labels

    return df
