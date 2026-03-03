# 🌿 Cotton Growth Stage (GSTD) Time-Series Risk Model

<p align="center">
  <img src="Images/banner.png" alt="Project Banner" width="900"/>
</p>

<p align="center">
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue"></a>
  <a href="#"><img alt="LightGBM" src="https://img.shields.io/badge/Model-LightGBM-brightgreen"></a>
  <a href="#"><img alt="Time Series" src="https://img.shields.io/badge/Type-Time%20Series-orange"></a>
  <a href="#"><img alt="Explainability" src="https://img.shields.io/badge/XAI-SHAP-purple"></a>
  <a href="#"><img alt="Status" src="https://img.shields.io/badge/Status-Capstone%20Project-informational"></a>
  <a href="#"><img alt="NDA Safe" src="https://img.shields.io/badge/Data-NDA--Safe%20Repo-red"></a>
</p>

---

## ✨ Overview

This repository contains an **end-to-end (NDA-safe)** machine learning pipeline for **predicting cotton growth stage (GSTD)** using **time-series agronomic + environmental features**.  
The output is both:

✅ **Predicted GSTD class** (multiclass classification)  
✅ **Interpretable risk score** (how uncertain / risky the prediction is)

The pipeline is designed for a real-world analytics client workflow and supports **run/treatment-aware splits** to prevent leakage.

---

## 🎯 Goals

- Predict **GSTD** from daily time-series signals
- Generate a **risk score** for decision support
- Provide **interpretability** (SHAP) to explain predictions
- Use **leakage-safe evaluation** (split by run/treatment rather than random)

---

## 🧠 Why LightGBM (Baseline)?

LightGBM is a strong baseline for tabular + engineered time-series features because it:

- Works extremely well with **lag + rolling-window features**
- Trains fast and supports early stopping
- Often beats deep learning when data is limited or noisy
- Supports explainability with **TreeSHAP**

> If needed, this repo can be extended to test LSTM/Transformers after baseline results.

---

## 🧩 Key Concepts (Simple)

### ⏳ Time-Series Feature Engineering
We transform raw daily signals into predictive features such as:

- **Lags**: `X(t-1), X(t-3), X(t-7), ...`
- **Rolling statistics**: `mean/std over last 7 or 14 days`
- **Deltas**: `X(t) - X(t-1)`

These capture *momentum, trend, and recent history* that helps infer growth stage.

### ⚠️ Risk Score
Two supported definitions:

- **1 − P(true class)** (high risk = low confidence in true class)
- **Sum of P(undesired stages)** (if you define “undesired” stages)

---

## 🗂️ Repository Structure


```text
cotton-gstd-timeseries-risk-model/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ Images/
│  └─ banner.png
├─ data/                     # ✅ local only (ignored by git)
│  └─ raw/
│     └─ your_dataset.csv
├─ notebooks/
│  └─ CAPSTONE_PROJECT.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ config.py              # project settings (columns, lags, horizon, CV, recall target)
│  ├─ data.py                # load + basic preprocessing + duplicate-column helper
│  ├─ labels.py              # y_trans_7 labeling (transition within horizon)
│  ├─ features.py            # lag/delta/rolling features (within RUNNO)
│  ├─ modeling.py            # LightGBM training + GroupKFold CV (RUNNO)
│  ├─ thresholding.py        # PR-curve table + threshold selection for target recall
│  └─ run_pipeline.py        # end-to-end: load → label → features → train → threshold → eval
└─ reports/                  # optional outputs (ignored by git)
   ├─ metrics/
   └─ figures/

```

<p align="center">
  <img src="Images/banner2.png" alt="Project Banner" width="900"/>
</p>

## 🚀 Quickstart
### 1) Create environment + install
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

```
### 2) Put your dataset locally (not committed)
data/raw/your_dataset.csv

This repo is NDA-safe: data/ is gitignored by default.

### 3) Run training pipeline
```bash
python -m src.pipeline.train --config src/config/default.yaml
```

## 📦 Outputs

After training, you’ll get:

## 📊 Metrics (JSON)

- reports/metrics_valid.json

- reports/metrics_test.json

### Includes:

- Accuracy

- Macro-F1

- Confusion matrix

### 🧾 Predictions (CSV)

- reports/predictions_valid.csv

- reports/predictions_test.csv

### Contains:

- true label

- predicted label

- class probabilities

- risk score


## 🔒 NDA / Data Safety

✅ What is safe to commit:

- code (src/)

- configs (src/config/)

- documentation (docs/)


🚫 What should NOT be committed:

- raw client data (data/)

## 🧪 Evaluation Strategy (No Leakage)

- Random splitting can leak time information.

- This repo supports:

- Split by RUNNO (recommended)

## 🗺️ Roadmap

 - LightGBM baseline with lag/rolling features

 - Leakage-safe run-based evaluation

 - Risk score output

 - SHAP plots saved to reports/figures/

 - Add LSTM comparison notebook

 - Add calibration (reliability curve)

 - Add experiment tracking (MLflow)


## 📄 License

Capstone / educational use.
