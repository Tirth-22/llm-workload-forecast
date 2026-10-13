"""
19_baseline_forecasters.py

Research validation module for the short-term LLM workload forecasting
pipeline.

Purpose
-------
Compare the existing HGB / CP-HGB forecasters against strong, transparent
forecasting baselines under the SAME chronological split.

Models evaluated
----------------
1. Persistence
2. Seasonal Naive (daily, 288 five-minute bins)
3. Moving Average (12 bins / 60 minutes)
4. Linear Regression
5. Random Forest
6. Existing HGB
7. Existing CP-HGB

Design rules
------------
- Exact Paper-3 chronological boundaries are preserved.
- No random split.
- No test-set training.
- No hyperparameter tuning on the test set.
- Baseline models use the same 24 non-change-point workload/calendar
  predictors used by HGB.
- Existing HGB and CP-HGB predictions are loaded from Module 05 output.
- The first 288 training rows are excluded because the existing feature
  definition requires the same maximum history.
- Forecast target is target_next_5min = workload at t+1.

Outputs
-------
results/predictions/baseline_forecast_predictions.csv
results/tables/baseline_forecast_metrics.csv
results/validation/baseline_forecasters_metadata.json
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "workload_features_5min.csv"

EXISTING_PREDICTIONS_FILE = (
    ROOT / "results" / "predictions" / "workload_forecast_predictions.csv"
)

PREDICTION_DIR = ROOT / "results" / "predictions"
TABLE_DIR = ROOT / "results" / "tables"
VALIDATION_DIR = ROOT / "results" / "validation"


# ============================================================
# EXACT EXISTING PAPER-3 PARTITIONS
# ============================================================

TRAIN_START = 0
TRAIN_END = 12383

CALIBRATION_START = 12384
CALIBRATION_END = 14975

TEST_START = 14976
TEST_END = 17565

MAX_HISTORY = 288

TARGET_COLUMN = "target_next_5min"
WORKLOAD_COLUMN = "total_tokens"
INTERVAL_COLUMN = "interval"

SEED = 42


# ============================================================
# SAME 24 HGB FEATURES AS MODULE 05
# ============================================================

HGB_FEATURES = [
    *(f"lag_{x}" for x in [1, 2, 3, 6, 12, 24, 72, 144, 288]),
    *(f"rolling_mean_{x}" for x in [3, 12, 72, 288]),
    *(f"rolling_std_{x}" for x in [3, 12, 72, 288]),
    *(f"rolling_max_{x}" for x in [12, 72, 288]),
    "daily_sin",
    "daily_cos",
    "weekly_sin",
    "weekly_cos",
]


# ============================================================
# METRICS
# ============================================================

def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def wape(actual, predicted):
    denominator = np.sum(np.abs(actual))
    if denominator == 0:
        return np.nan
    return float(np.sum(np.abs(actual - predicted)) / denominator * 100.0)


def smape(actual, predicted):
    denominator = np.abs(actual) + np.abs(predicted)
    valid = denominator > 0
    if not np.any(valid):
        return np.nan
    return float(
        np.mean(
            2.0
            * np.abs(actual[valid] - predicted[valid])
            / denominator[valid]
        )
        * 100.0
    )


def mean_bias(actual, predicted):
    return float(np.mean(predicted - actual))


def calculate_metrics(actual, predicted, model_name):
    return {
        "model": model_name,
        "test_rows": int(len(actual)),
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "WAPE_percent": wape(actual, predicted),
        "sMAPE_percent": smape(actual, predicted),
        "mean_bias": mean_bias(actual, predicted),
    }


# ============================================================
# VALIDATION
# ============================================================

def require_columns(df, columns, name):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )


def create_partitions(df):
    train = df[
        (df[INTERVAL_COLUMN] >= TRAIN_START)
        & (df[INTERVAL_COLUMN] <= TRAIN_END)
    ].copy()

    calibration = df[
        (df[INTERVAL_COLUMN] >= CALIBRATION_START)
        & (df[INTERVAL_COLUMN] <= CALIBRATION_END)
    ].copy()

    test = df[
        (df[INTERVAL_COLUMN] >= TEST_START)
        & (df[INTERVAL_COLUMN] <= TEST_END)
    ].copy()

    return train, calibration, test


def validate_chronology(train, calibration, test):
    if not (
        int(train[INTERVAL_COLUMN].iloc[-1])
        < int(calibration[INTERVAL_COLUMN].iloc[0])
        and int(calibration[INTERVAL_COLUMN].iloc[-1])
        < int(test[INTERVAL_COLUMN].iloc[0])
    ):
        raise RuntimeError("Chronological partition validation failed.")


def validate_feature_history(df):
    feature_nan_count = int(df[HGB_FEATURES].isna().sum().sum())

    # The first 288 rows are expected to lack history.
    expected_warmup_rows = MAX_HISTORY

    if feature_nan_count == 0:
        raise RuntimeError(
            "Expected warm-up feature NaNs were not found. "
            "Refusing to silently change the existing feature definition."
        )

    train = df[
        (df[INTERVAL_COLUMN] >= TRAIN_START)
        & (df[INTERVAL_COLUMN] <= TRAIN_END)
    ].copy()

    train_modelled = train[
        ~train[HGB_FEATURES].isna().any(axis=1)
    ].copy()

    removed = len(train) - len(train_modelled)

    if removed != expected_warmup_rows:
        raise RuntimeError(
            "Unexpected number of training warm-up rows removed: "
            f"expected {expected_warmup_rows}, found {removed}."
        )


# ============================================================
# CAUSAL BASELINES
# ============================================================

def build_causal_baselines(df):
    """
    Build forecasts using only workload values available at forecast origin t.

    At row t:
        target_next_5min = x[t+1]

    Therefore:
        persistence      = x[t]
        daily naive      = x[t-287]
        moving average   = mean(x[t], ..., x[t-11])

    The daily naive uses 288 five-minute bins = 24 hours.
    """

    result = df[
        [INTERVAL_COLUMN, TARGET_COLUMN, WORKLOAD_COLUMN]
    ].copy()

    result["persistence_prediction"] = (
        result[WORKLOAD_COLUMN]
    )

    result["seasonal_naive_prediction"] = (
        result[WORKLOAD_COLUMN].shift(MAX_HISTORY - 1)
    )

    result["moving_average_prediction"] = (
        result[WORKLOAD_COLUMN]
        .rolling(window=12, min_periods=12)
        .mean()
    )

    return result


# ============================================================
# MACHINE-LEARNING BASELINES
# ============================================================

def train_ml_baselines(train):
    train_modelled = train[
        ~train[HGB_FEATURES].isna().any(axis=1)
        & train[TARGET_COLUMN].notna()
    ].copy()

    X_train = train_modelled[HGB_FEATURES]
    y_train = train_modelled[TARGET_COLUMN].astype(float)

    if X_train.isna().any().any():
        raise RuntimeError("ML baseline training features contain NaNs.")

    if not np.isfinite(y_train.to_numpy()).all():
        raise RuntimeError("ML baseline target contains non-finite values.")

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

    return models


# ============================================================
# EXISTING MODEL RESULTS
# ============================================================

def load_existing_model_predictions():
    if not EXISTING_PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            "Existing Module 05 prediction file not found:\n"
            f"{EXISTING_PREDICTIONS_FILE}"
        )

    existing = pd.read_csv(EXISTING_PREDICTIONS_FILE)

    require_columns(
        existing,
        [
            INTERVAL_COLUMN,
            "split",
            "actual_next_5min_tokens",
            "hgb_prediction",
            "cp_hgb_prediction",
        ],
        "Existing Module 05 predictions",
    )

    return existing


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_prediction_vector(prediction, name):
    values = np.asarray(prediction, dtype=float)

    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} contains non-finite values.")

    if (values < 0).any():
        raise RuntimeError(f"{name} contains negative predictions.")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("MODULE 19 — BASELINE FORECASTERS")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input feature matrix not found:\n{INPUT_FILE}\n\n"
            "Run Modules 01-04 first."
        )

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. LOAD
    # --------------------------------------------------------

    print("\n[1] Loading existing workload feature matrix...")
    df = pd.read_csv(INPUT_FILE)

    require_columns(
        df,
        [
            INTERVAL_COLUMN,
            WORKLOAD_COLUMN,
            TARGET_COLUMN,
            *HGB_FEATURES,
        ],
        "Workload feature matrix",
    )

    if len(df) != TEST_END + 1:
        raise RuntimeError(
            "Unexpected row count. "
            f"Expected {TEST_END + 1}, found {len(df)}."
        )

    # --------------------------------------------------------
    # 2. VALIDATE CAUSAL FEATURE HISTORY
    # --------------------------------------------------------

    print("\n[2] Validating existing feature history...")
    validate_feature_history(df)

    # --------------------------------------------------------
    # 3. EXACT PARTITIONS
    # --------------------------------------------------------

    print("\n[3] Creating exact chronological partitions...")
    train, calibration, test = create_partitions(df)
    validate_chronology(train, calibration, test)

    test_modelled = test[test[TARGET_COLUMN].notna()].copy()

    if len(test_modelled) != 2589:
        raise RuntimeError(
            "Unexpected usable test size. "
            f"Expected 2589, found {len(test_modelled)}."
        )

    # --------------------------------------------------------
    # 4. CAUSAL BASELINES
    # --------------------------------------------------------

    print("\n[4] Building causal statistical baselines...")
    baseline_frame = build_causal_baselines(df)

    # Align by interval rather than relying on row order.
    test_baselines = test_modelled[
        [INTERVAL_COLUMN, TARGET_COLUMN]
    ].merge(
        baseline_frame,
        on=[INTERVAL_COLUMN, TARGET_COLUMN],
        how="left",
        validate="one_to_one",
    )

    baseline_prediction_columns = {
        "Persistence": "persistence_prediction",
        "Seasonal Naive (Daily)": "seasonal_naive_prediction",
        "Moving Average (60 min)": "moving_average_prediction",
    }

    # --------------------------------------------------------
    # 5. ML BASELINES
    # --------------------------------------------------------

    print("\n[5] Training ML baselines...")
    ml_models = train_ml_baselines(train)

    X_test = test_modelled[HGB_FEATURES]

    if X_test.isna().any().any():
        raise RuntimeError("Test feature matrix contains NaNs.")

    for name, model in ml_models.items():
        prediction = model.predict(X_test)
        prediction = np.clip(prediction, 0.0, None)
        test_baselines[name] = prediction

    # --------------------------------------------------------
    # 6. LOAD EXISTING HGB / CP-HGB
    # --------------------------------------------------------

    print("\n[6] Loading existing HGB and CP-HGB predictions...")
    existing = load_existing_model_predictions()

    existing_test = existing[
        existing["split"].astype(str).str.lower() == "test"
    ].copy()

    existing_test = existing_test[
        existing_test[TARGET_COLUMN.replace(
            "target_next_5min", "actual_next_5min_tokens"
        )].notna()
    ].copy()

    # The existing output uses actual_next_5min_tokens rather than
    # target_next_5min. Rename only for alignment.
    existing_test = existing_test.rename(
        columns={
            "actual_next_5min_tokens": TARGET_COLUMN
        }
    )

    existing_test = existing_test[
        [INTERVAL_COLUMN, TARGET_COLUMN, "hgb_prediction", "cp_hgb_prediction"]
    ]

    merged = test_baselines.merge(
        existing_test,
        on=[INTERVAL_COLUMN, TARGET_COLUMN],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(test_modelled):
        raise RuntimeError(
            "Existing HGB/CP-HGB predictions do not align exactly "
            "with the locked test set."
        )

    # --------------------------------------------------------
    # 7. FINAL MODEL SET
    # --------------------------------------------------------

    prediction_columns = {
        **baseline_prediction_columns,
        "Linear Regression": "Linear Regression",
        "Random Forest": "Random Forest",
        "HGB": "hgb_prediction",
        "CP-HGB": "cp_hgb_prediction",
    }

    actual = merged[TARGET_COLUMN].astype(float).to_numpy()

    metrics_rows = []

    for model_name, column in prediction_columns.items():
        prediction = merged[column].astype(float).to_numpy()
        validate_prediction_vector(prediction, model_name)

        metrics_rows.append(
            calculate_metrics(
                actual,
                prediction,
                model_name,
            )
        )

    metrics_df = pd.DataFrame(metrics_rows)

    # --------------------------------------------------------
    # 8. SAVE PREDICTIONS
    # --------------------------------------------------------

    print("\n[7] Saving locked-test baseline predictions...")

    output_prediction_columns = [
        INTERVAL_COLUMN,
        TARGET_COLUMN,
        *prediction_columns.values(),
    ]

    output_predictions = merged[
        output_prediction_columns
    ].copy()

    output_predictions = output_predictions.rename(
        columns={
            TARGET_COLUMN: "actual_next_5min_tokens",
            "persistence_prediction": "persistence_prediction",
            "seasonal_naive_prediction": "seasonal_naive_prediction",
            "moving_average_prediction": "moving_average_prediction",
        }
    )

    prediction_output = (
        PREDICTION_DIR / "baseline_forecast_predictions.csv"
    )

    output_predictions.to_csv(
        prediction_output,
        index=False,
    )

    # --------------------------------------------------------
    # 9. SAVE METRICS
    # --------------------------------------------------------

    metrics_output = TABLE_DIR / "baseline_forecast_metrics.csv"
    metrics_df.to_csv(metrics_output, index=False)

    # --------------------------------------------------------
    # 10. SAVE METADATA
    # --------------------------------------------------------

    metadata = {
        "module": "19_baseline_forecasters",
        "purpose": (
            "Fair locked-test comparison of transparent forecasting "
            "baselines against existing HGB and CP-HGB."
        ),
        "dataset": "BurstGPT_1",
        "target": TARGET_COLUMN,
        "forecast_horizon": "next five-minute interval",
        "train_partition": [TRAIN_START, TRAIN_END],
        "calibration_partition": [CALIBRATION_START, CALIBRATION_END],
        "locked_test_partition": [TEST_START, TEST_END],
        "usable_locked_test_rows": int(len(merged)),
        "maximum_history": MAX_HISTORY,
        "models": list(prediction_columns.keys()),
        "ml_features": HGB_FEATURES,
        "moving_average_window_bins": 12,
        "seasonal_period_bins": MAX_HISTORY,
        "random_split": False,
        "test_used_for_training": False,
        "test_used_for_hyperparameter_tuning": False,
        "existing_hgb_predictions_reused": True,
        "existing_cp_hgb_predictions_reused": True,
    }

    metadata_output = (
        VALIDATION_DIR / "baseline_forecasters_metadata.json"
    )

    with open(metadata_output, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    # --------------------------------------------------------
    # 11. PRINT FINAL TABLE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODULE 19 RESULT")
    print("=" * 70)

    print(
        metrics_df[
            [
                "model",
                "MAE",
                "RMSE",
                "WAPE_percent",
                "sMAPE_percent",
                "mean_bias",
            ]
        ].to_string(index=False)
    )

    print("\nCreated:")
    print(f"  {prediction_output}")
    print(f"  {metrics_output}")
    print(f"  {metadata_output}")

    print("\nNo model was declared superior automatically.")
    print("Locked test data was used only for final evaluation.")
    print("MODULE 19 COMPLETE")


if __name__ == "__main__":
    main()
