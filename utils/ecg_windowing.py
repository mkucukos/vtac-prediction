import pandas as pd
import os
import numpy as np
import wfdb


def make_baseline_windows(
    healthy_subject, sample_rate=250, win_sec=30, shift_sec=5, max_subjects=100
):
    WINDOW_SIZE = win_sec * sample_rate
    WINDOW_SHIFT = shift_sec * sample_rate
    out = []

    for _, row in healthy_subject.iterrows():
        sid, ecg = row["subject_id"], row["ecg_data"]
        if not isinstance(ecg, list) or len(ecg) < WINDOW_SIZE:
            continue
        for start in range(0, len(ecg) - WINDOW_SIZE + 1, WINDOW_SHIFT):
            end = start + WINDOW_SIZE
            out.append(
                {
                    "Record": sid,
                    "Start": start,
                    "End": end,
                    "Label": "baseline",
                    "ECG": ecg[start:end],
                }
            )

    df = pd.DataFrame(out)
    if max_subjects is not None:
        keep = df["Record"].unique()[:max_subjects]
        df = df[df["Record"].isin(keep)].copy()
    return df


def window_vtac_records(
    record_dir: str,
    sample_rate: int = 250,
    win_sec: int = 30,
    shift_sec: int = 5,
    lead: int = 0,
    ann_ext: str = "atr",
) -> pd.DataFrame:
    """Create sliding windows and label as VTAC / Pre-VTAC / Other."""
    WIN = win_sec * sample_rate
    STRIDE = shift_sec * sample_rate

    record_files = [f for f in os.listdir(record_dir) if f.endswith(".dat")]
    record_names = [os.path.splitext(f)[0] for f in record_files]

    rows = []
    for name in record_names:
        try:
            path = os.path.join(record_dir, name)
            sig, _ = wfdb.rdsamp(path)
            ann = wfdb.rdann(path, ann_ext)
            ecg = sig[:, lead]

            starts = [s for s, sym in zip(ann.sample, ann.symbol) if sym == "["]
            ends = [s for s, sym in zip(ann.sample, ann.symbol) if sym == "]"]
            intervals = []
            for s in starts:
                after = [e for e in ends if e > s]
                intervals.append((s, after[0] if after else s + 60 * sample_rate))

            N = len(ecg)
            for st in range(0, N - WIN + 1, STRIDE):
                en = st + WIN
                seg = ecg[st:en]

                overlaps = any((st < e and en > s) for s, e in intervals)
                if overlaps:
                    label = "VTAC"
                else:
                    next_st = min([s for s, _ in intervals if s >= st], default=None)
                    label = (
                        "Pre-VTAC"
                        if (next_st is not None and en <= next_st)
                        else "Other"
                    )

                rows.append(
                    {"Record": name, "Start": st, "End": en, "Label": label, "ECG": seg}
                )
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}")

    return pd.DataFrame(rows)
