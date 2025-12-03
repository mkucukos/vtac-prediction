# VTAC Prediction Pipeline

Lightweight end-to-end workflow for **VTAC (ventricular tachyarrhythmia) prediction** using ECG windowing, feature extraction (QT/TMV/QRS/ST), z-scoring, and ML models (RF/XGBoost) with **GroupKFold** subject splits.

## Contents
- Preprocessing (windowing, labeling, feature extraction, z-scoring)
- Model improvement (feature sets, tuning, CV diagnostics)
- Testing & validation (held-out evaluation, plots)

## Project Structure

```bash
├── notebooks/
│ ├── 01_preprocessing.ipynb
│ ├── 02_model_improvement.ipynb
│ └── 03_model_testing_validation.ipynb
├── utils/
│ ├── ecg_windowing.py # window_vtac_records, make_baseline_windows
│ ├── ecg_features.py # filters, TMV/QT/QRS/ST, process_dataframe
│ └── ecg_plots.py # plot_tmv_qt_per_subject and other visuals
├── model/ # saved estimators, feature lists, thresholds
└── README.md
```

## Setup
```python
# Python 3.11 recommended
python -m venv .venv
source .venv/bin/activate     # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```


## Quick Start

### 1 Preprocessing
Run `notebooks/01_preprocessing.ipynb` to:
- Read WFDB records from `data/raw/`
- Create sliding windows & labels (`utils/ecg_windowing.py`, `utils/vtac_labeling.py`)
- Extract ECG features (`utils/ecg_features.py`)
- Build per-subject baselines & z-scores
- Save outputs to `data/processed/`:
  - `windowed.parquet`, `features.parquet`, `zscores.parquet`
  - Optional: `zscores_df.pkl` (pandas pickle)

**Save a pickle with pandas:**
```python
import os, pandas as pd
os.makedirs("data/processed", exist_ok=True)
zscores_df.to_pickle("data/processed/zscores_df.pkl")
# Load: pd.read_pickle("data/processed/zscores_df.pkl")
```

### 2 Model Improvement
Run `notebooks/02_model_improvement.ipynb` to:
- Define feature sets (incl. TMV/QT/QRS/ST z-scores)
- Tune models with `RandomizedSearchCV`
- Use `GroupKFold` by subject
- Inspect metrics & feature importance
- Save artifacts to `models/`:
  - `best_model.joblib`, `feature_list.json`, `thresholds.json`

### 3 Testing & Validation
Run `notebooks/03_model_testing_validation.ipynb` to:
- Load locked artifacts and evaluate on held-out subjects
- Report ROC-AUC, PR-AUC, F1, precision/recall
- Plot probabilities vs VTAC timeline
- Export figures/tables for manuscripts

## Utilities (import examples)
```python
from utils.ecg_windowing import window_vtac_records, make_baseline_windows
from utils.vtac_labeling import extend_last_vtac_label_inplace
from utils.ecg_features import process_dataframe, calculate_tmv_and_qt
from utils.ecg_plots import plot_tmv_qt_per_subject
```

## Notes
- Sampling rate default: **250 Hz**; window **30 s**; shift **5 s**.
- VTAC intervals inferred from WFDB annotations (`[`, `]`).
- Baselines are subject-specific (median references) and z-scoring uses **baseline-only** stats.

## Feature Glossary

### 🔹 TMV_Score — Local (Within-Window) T-Wave Variability

**Definition:**  
Variability of T-wave shapes **within a single 30-second window**, computed as the **mean squared deviation** between each beat’s T-wave and the window-averaged T-wave.

**What it captures:**  
Beat-to-beat morphological variability — short-term T-wave instability.

**Use case:**  
Flags **acute/transient repolarization instability**, potentially predictive of **imminent arrhythmic events**.

**Key takeaway:** ✅ Captures **local** morphological instability **within the window**.


### 🔹 TMV_Global — Deviation from Subject-Specific Reference T-Wave

**Definition:**  
**Mean squared error** between the **current window’s averaged T-wave** and a **subject-specific reference T-wave** (e.g., the median T-wave from baseline/pre-VTAC periods).

**What it captures:**  
Degree of deviation from a subject’s **normal repolarization pattern** over time, independent of short-term fluctuations.

**Use case:**  
Tracks **gradual/sustained changes** (e.g., ischemic drift, progressive repolarization abnormality).

**Key takeaway:** ✅ Captures **global** abnormality **relative to subject baseline**.


## Reproducibility
- Set `random_state=42` in all model/tuning steps.
- Document excluded subjects and feature lists in the model notebooks.

## Licensing
This project is licensed under the MIT License — see the LICENSE