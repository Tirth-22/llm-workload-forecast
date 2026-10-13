# LLM Adaptive Research

This repository implements a research pipeline for adaptive cloud capacity management of large-language-model (LLM) inference workloads. The project studies short-term workload forecasting, failure-risk prediction, change-point detection, and capacity-control policy evaluation using BurstGPT workload traces.

The implementation is organized as a numbered experiment pipeline in `src/`, with trained models stored under `models/` and generated results under `results/`.

## Overview

The project follows a multi-stage research workflow:

1. Validate and preprocess raw BurstGPT workload data.
2. Detect change points and build causal workload features.
3. Train short-term forecasting models for 5-minute workload prediction.
4. Train a failure-risk classifier for request failure probability estimation.
5. Calibrate raw risk probabilities and build adaptive threshold controllers.
6. Simulate capacity policies and compare performance under normalized capacity assumptions.
7. Evaluate forecast, risk, and capacity metrics on locked test data.
8. Freeze and audit the final implementation outputs.

## Repository structure

```text
llm-adaptive-research/
├── data/                          # Raw and derived datasets
│   ├── BurstGPT_1.csv
│   ├── BurstGPT_2.csv
│   ├── burstgpt_5min.csv
│   ├── burstgpt_minute.csv
│   ├── failure_features_minute.csv
│   └── workload_features_5min.csv
├── models/                        # Trained model artifacts
│   ├── calibration/
│   ├── failure_risk/
│   └── workload_forecaster/
├── results/                      # Generated predictions, tables, validation, and policies
│   ├── ablation/
│   ├── alarms/
│   ├── figures/
│   ├── policies/
│   ├── predictions/
│   ├── sensitivity/
│   ├── tables/
│   └── validation/
├── src/                          # Research pipeline modules
│   ├── 01_data_validation.py
│   ├── 02_build_time_views.py
│   ├── 03_change_point_detector.py
│   ├── 04_build_workload_features.py
│   ├── 05_train_short_term_forecaster.py
│   ├── 06_build_failure_features.py
│   ├── 07_train_failure_risk.py
│   ├── 08_calibration.py
│   ├── 09_adaptive_threshold.py
│   ├── 10_capacity_simulation.py
│   ├── 11_change_point_reset.py
│   ├── 12_metrics.py
│   ├── 13_bootstrap_tests.py
│   ├── 14_make_figures.py
│   ├── 15_make_tables.py
│   ├── 16_ablation_risk_features.py
│   ├── 17_change_point_sensitivity.py
│   ├── 18_change_point_downstream_robustness.py
│   ├── 19_baseline_forecasters.py
│   ├── 20_forecast_to_control_comparison.py
│   ├── 21_forecast_adaptive_risk_control.py
│   ├── 22_paired_policy_statistics.py
│   ├── 23_final_implementation_audit.py
│   └── ...
├── .gitignore
├── .venv/
├── README.md
└── ...
```

## Main research components

### 1. Workload forecasting

The research builds causal forecasting features using:

- lagged token demand values
- shifted rolling means and standard deviations
- rolling max values
- calendar features (daily and weekly periodicity)
- change-point statistics such as score, alarm, and short/long ratio

The forecasting stage trains two short-term models:

- HGB (HistGradientBoostingRegressor)
- CP-HGB (change-point-aware HGB variant)

These models predict the next 5-minute workload in token units.

### 2. Failure-risk prediction

The failure-risk model uses minute-level causal features derived from workload and service metrics, including:

- request count
- completed tokens
- GPT-4 share
- API share
- zero-response count
- lag and rolling statistics for each source series
- minute-of-day and day-of-week cyclic features

The classifier is implemented using XGBoost and targets a binary failure outcome.

### 3. Calibration and adaptive thresholding

Raw model probabilities are calibrated to better reflect true failure likelihood. The implementation then applies an adaptive threshold controller to convert probability estimates into operational decisions, while keeping the threshold within allowed bounds.

### 4. Capacity simulation and policy evaluation

The project compares several capacity-control policies:

- V1: forecast-only policy
- V2: fixed-risk policy
- V3: adaptive-risk policy
- V4: adaptive reset policy

These policies are evaluated according to:

- under-provisioned percentage
- mean shortage
- p95 shortage
- CVaR95 shortage
- mean unused capacity
- reserve activation rate
- cost index
- total shortage and unused capacity

## Implementation flow

The pipeline is intentionally ordered and should be executed in sequence.

### Core execution order

```bash
python src/01_data_validation.py
python src/02_build_time_views.py
python src/03_change_point_detector.py
python src/04_build_workload_features.py
python src/05_train_short_term_forecaster.py
python src/06_build_failure_features.py
python src/07_train_failure_risk.py
python src/08_calibration.py
python src/09_adaptive_threshold.py
python src/10_capacity_simulation.py
python src/11_change_point_reset.py
python src/12_metrics.py
python src/13_bootstrap_tests.py
python src/14_make_figures.py
python src/15_make_tables.py
python src/16_ablation_risk_features.py
python src/17_change_point_sensitivity.py
python src/18_change_point_downstream_robustness.py
python src/19_baseline_forecasters.py
python src/20_forecast_to_control_comparison.py
python src/21_forecast_adaptive_risk_control.py
python src/22_paired_policy_statistics.py
python src/23_final_implementation_audit.py
```

Notes:

- These modules are designed to be run from the project root.
- Some modules expect prior outputs to exist in `results/` and `models/`.
- The project uses chronological, non-random splits and keeps locked test data separate from model development.

## Dependencies

This project relies on standard scientific Python libraries, including:

- Python 3.10+
- NumPy
- pandas
- scikit-learn
- XGBoost
- joblib
- matplotlib
- seaborn

For Windows PowerShell, the virtual environment can be activated with:

```powershell
.\.venv\Scripts\Activate.ps1
```

If needed, install the required packages with:

```bash
pip install numpy pandas scikit-learn xgboost joblib matplotlib seaborn
```

## Outputs

### Model artifacts

Trained model files are stored in:

- `models/workload_forecaster/`
- `models/failure_risk/`
- `models/calibration/`

### Prediction outputs

The research stores prediction tables in:

- `results/predictions/`

Examples include:

- workload forecast predictions
- calibrated failure-risk predictions
- policy simulation outputs

### Evaluation and analysis outputs

The final evaluation tables are saved in:

- `results/tables/`
- `results/validation/`
- `results/figures/`
- `results/policies/`

Key output files include:

- `forecast_metrics.csv`
- `risk_metrics.csv`
- `capacity_metrics.csv`
- `evaluation_metrics.json`
- final policy comparison tables
- freeze/audit metadata generated by Module 23

## Evaluation methodology

Module 12 (`src/12_metrics.py`) performs the principal locked-test evaluation for:

- workload forecasting metrics (MAE, RMSE, WAPE, sMAPE, mean bias)
- failure-risk metrics (ROC-AUC, average precision, calibrated Brier, ECE, precision, recall, F1)
- capacity-policy metrics (shortage, under-provisioning, unused capacity, reserve activation, cost index)

This module explicitly uses locked test partitions only and does not retrain or tune models after the evaluation split is fixed.

## Research goal

The implementation supports a research objective for LLM-systems operations: balancing performance and capacity under dynamic workload spikes, while using predictive risk signals to trigger adaptive capacity policy responses.

In short, the project models how an LLM serving system can forecast near-term demand, anticipate failure risk, and make adaptive provisioning decisions with a controlled and measured policy response.

## Notes

- The raw dataset is not modified during validation.
- The project uses chronological partitions instead of random splits for realistic time-series evaluation.
- The final audit stage verifies that artifacts, schemas, and counts are consistent before research freeze.
- The repository is designed for reproducible research and experiment tracking, not just one-off training scripts.

## License

This project does not currently declare a license in the repository root. If you are using or publishing this code, verify the intended licensing terms before redistribution.

## Contact / usage

This implementation is intended for research and experimental studies. For production use or deployment, additional validation, packaging, and operational hardening would be required.
