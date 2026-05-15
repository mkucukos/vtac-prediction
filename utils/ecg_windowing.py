import pandas as pd
import os
import numpy as np
import wfdb

from .ecg_features import clipping_ratio, flatline_ratio


def window_vtac_records(
    record_dir: str,
    sample_rate: int = 250,
    win_sec: int = 30,
    shift_sec: int = 5,
    lead: int = 0,
    ann_ext: str = "atr",
    mad_k: float = 8.0,
    max_clip_ratio: float = 0.15,
    verbose: bool = False,
) -> pd.DataFrame:
    """Create sliding windows, label as VTAC / Pre-VTAC / Other, and apply QC."""
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

                clip_ratio = clipping_ratio(seg, k=mad_k)[1]
                flat_flag  = flatline_ratio(seg)
                qc_pass    = (clip_ratio <= max_clip_ratio) and (flat_flag == 0.0)

                if verbose and not qc_pass:
                    reasons = []
                    if clip_ratio > max_clip_ratio:
                        reasons.append(f"clip_ratio={clip_ratio:.2f}")
                    if flat_flag == 1.0:
                        reasons.append("flatline")
                    print(f"[QC][{name}] window @{st} | {', '.join(reasons)}")

                rows.append({
                    "Record": name, "Start": st, "End": en, "Label": label,
                    "ECG": seg, "Clip_Ratio": clip_ratio,
                    "Flatline_Flag": flat_flag, "QC_Pass": qc_pass,
                })
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}")

    return pd.DataFrame(rows)
