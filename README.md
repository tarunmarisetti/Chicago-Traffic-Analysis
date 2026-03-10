# Chicago Traffic Crash Severity Prediction

> Predicting severe crash outcomes from 911K+ Chicago crash records using LightGBM, 
> threshold optimization, and SHAP explainability.

## Overview

This is a capstone data science project (FSE 570, Arizona State University, Fall 2025) 
analyzing traffic crash data from the City of Chicago. The project covers 4 analysis 
components — this repo focuses primarily on the **predictive modeling component** for 
crash injury severity classification.

**The core challenge:** Only 1.66% of crashes are severe — making this a highly 
imbalanced binary classification problem where accuracy is meaningless and recall is 
everything.

---

## My Contributions

- Built and evaluated the full ML pipeline for crash severity prediction
- Compared baseline models (DummyClassifier, Logistic Regression, Random Forest) 
  against LightGBM on ~950K records
- Addressed class imbalance via class weighting and oversampling strategies
- Implemented F₂-score based decision threshold tuning to maximize recall for 
  severe crashes
- Conducted cost-sensitive threshold analysis simulating real-world emergency 
  response priorities
- Performed SHAP-based model explainability analysis

---

## Dataset

| Source | Records | Columns | Period |
|---|---|---|---|
| City of Chicago — Traffic Crash Records | 911K rows | 48 | Sep 2017 – Oct 2025 |
| CoC Red Light & Speed Camera Violations | 1.5M rows | 9 | 2014 – 2025 |
| CoC Camera Locations | 360 rows | 8 | 2004 – 2025 |
| US Census Bureau ACS 5-Year | 61 rows | 6 | 2023 |

Data source: [Chicago Data Portal](https://data.cityofchicago.org/Transportation/)

---

## Problem Definition

**Target variable:** Binary — `severe` (fatal or incapacitating injury) vs `non-severe`  
**Class distribution:** 1.66% severe / 98.34% non-severe  
**Primary metric:** PR-AUC and F₂-score (prioritizes recall over precision)

Standard accuracy is useless here — a model predicting "non-severe" for everything 
gets 98.4% accuracy but catches zero severe crashes.

---

## Models & Results

| Model | PR-AUC | Notes |
|---|---|---|
| DummyClassifier (baseline) | ~0.016 | Predicts majority class only |
| Logistic Regression | ~0.146 | Balanced class weights, one-hot encoding |
| Random Forest | ~0.134 | Balanced weights, higher compute cost |
| **LightGBM (final)** | **~0.153** | Best overall, fast on 950K rows |

### Key Finding — Threshold Tuning Was the Game Changer

At default threshold (0.50):
- Recall for severe crashes ≈ **0%** — completely useless in safety context

After F₂-based threshold tuning (threshold = 0.061):
- Recall for severe crashes → **55%+**
- Precision for severe cases → **12–13%**
- Practical outcome: model captures majority of severe crashes with manageable 
  false alarms — useful for traffic analysts and emergency services

---

## Why LightGBM

- Handles large tabular data with mixed numerical/categorical features natively
- Leaf-wise tree growth outperforms level-wise on this data type
- Built-in categorical feature support — no manual one-hot encoding needed
- Fast training on ~950K rows
- Hyperparameter tuning showed minimal gains over defaults — defaults are 
  already near-optimal; class imbalance is the primary bottleneck, not model capacity

---

## Class Imbalance Strategies Tested

| Strategy | Result |
|---|---|
| No weighting (baseline) | Best overall PR-AUC |
| `class_weight="balanced"` | Similar PR-AUC, slight score shift |
| Oversampling (10–30% severe ratio) | No improvement, added noise |

**Conclusion:** Original unsampled dataset with threshold tuning outperformed all 
resampling strategies.

---

## SHAP Explainability

Top features driving severity predictions (consistent across weighted/unweighted models):

- **`CRASH_TYPE`, `FIRST_CRASH_TYPE`** — crash mechanism is the strongest signal
- **`PRIM_CONTRIBUTORY_CAUSE`** — contributing cause has strong directional impact
- **`HIT_AND_RUN_I`, `NUM_UNITS`** — incident characteristics
- **`CRASH_HOUR`, `CRASH_MONTH`** — temporal patterns matter
- **`WEATHER_CONDITION`, `IS_NIGHT`** — environmental context
- **`POSTED_SPEED_LIMIT`, `ROAD_SURFACE`** — smaller but visible contributions

SHAP analysis confirmed the model learns logically sound patterns aligned with 
real-world traffic safety knowledge — not statistical artifacts.

---

## Tech Stack
```
Python · Pandas · Scikit-learn · LightGBM · SHAP · Matplotlib · Statsmodels
```

---

## Project Team

ASU FSE 570 — Fall 2025  
**Tarun Sai Marisetti** *(classification modeling)* ·
Lahiru Gunasekara · Dante Iannello · Meghana Kankanala · Saisrinivas Mamunuru 


---

## References

- City of Chicago Data Portal: https://data.cityofchicago.org
- US Census Bureau ACS: https://api.census.gov/data/2023/acs/acs5
