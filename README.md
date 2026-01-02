# VTAC Prediction Pipeline

<p align="center">
  <img src="assets/main.png" width="80%">
</p>

Lightweight end-to-end workflow for **VTAC (Ventricular Tachycardia) prediction** using ECG windowing, feature extraction (QT/TMV/QRS/ST), z-scoring, and ML models (RF/XGBoost) with **GroupKFold** subject splits.

---

## Contents
- Preprocessing (windowing, labeling, feature extraction, z-scoring)
- Model improvement (feature sets, tuning, CV diagnostics)
- Testing & validation (held-out evaluation, plots)

---

## Project Structure

```bash
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_model_improvement.ipynb
│   └── 03_model_testing_validation.ipynb
├── utils/
│   ├── ecg_windowing.py
│   ├── ecg_features.py
│   └── ecg_plots.py
├── model/
├── assets/
│   ├── main.png
│   ├── model_training.png
│   └── external_validation.png
└── README.md

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
- Build per-subject, real-time robust standardization
- Save outputs to `data/processed/`zscores_df.pkl` (pandas pickle)

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
  - `best_model.joblib`,

The figure below illustrates feature dynamics and model behavior during training:

<p align="center">
  <img src="assets/model_training.png" width="80%">
</p>

### 3 Testing & Validation
Run `notebooks/03_model_testing_validation.ipynb` to:
- Load locked artifacts and evaluate on held-out subjects
- Report ROC-AUC, PR-AUC, F1, precision/recall
- Plot probabilities vs VTAC timeline
- Export figures/tables for manuscripts

The following figure shows an example of predicted VTAC risk compared to ground truth:

<p align="center">
  <img src="assets/external_validation.png" width="80%">
</p>

## Utilities (import examples)
```python
from utils.ecg_windowing import window_vtac_records
from utils.ecg_features import process_dataframe, create_windowed_ecg_from_mat , convert_and_relabel_windowed_df_full
from utils.ecg_plots import compute_refs_and_zscores, plot_subject_panels 
```

## Notes
- Sampling rate default: **250 Hz**; window **30 s**; shift **5 s**.
- VTAC intervals inferred from WFDB annotations (`[`, `]`).
- Z-scoring uses IQR-based robust statistics, computed causally using past windows only.

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