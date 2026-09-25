"""
07_train_failure_risk.py

Train the Paper 4-style XGBoost failure-risk model.

Research:
Short-Term LLM Workload Prediction for Adaptive Cloud Computing

Paper 4 reproduced components:
- 89 causal failure-risk predictors
- Chronological partitioning
- XGBoost failure-risk classifier
- Paper 4 selected XGBoost hyperparameters

Important:
- Current experiment uses BurstGPT_1.csv only.
- Therefore Paper 4's original absolute minute boundaries
  cannot be reproduced exactly.
- We preserve the chronological role of the partitions
  using proportional boundaries over the available trace.
- Initial 1440 minutes are excluded because the largest
  causal history window is 1440 minutes.
- No random splitting.
- No calibration in this module.
- No test-set tuning.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "failure_features_minute.csv"

MODEL_DIR = ROOT / "models" / "failure_risk"
RESULT_DIR = ROOT / "results" / "predictions"
META_DIR = ROOT / "results" / "validation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

# Paper 4 selected XGBoost configuration
XGB_PARAMS = {
    "n_estimators": 350,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "reg_lambda": 5.0,
    "reg_alpha": 0.1,

    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": SEED,
    "n_jobs": -1,
}


# ============================================================
# PAPER 4 FEATURE DEFINITIONS
# ============================================================

SOURCE_SERIES = [
    "request_count",
    "completed_tokens",
    "gpt4_share",
    "api_share",
    "zero_response_count",
]

LAGS = [
    1,
    2,
    5,
    15,
    60,
    240,
    1440,
]

ROLLING_WINDOWS = [
    5,
    15,
    60,
    240,
    1440,
]

EXPECTED_FEATURE_COUNT = 89


# ============================================================
# BUILD EXACT 89 FEATURE NAMES
# ============================================================

def build_feature_names():
    """
    Paper 4:

    Five source series.

    Each source:
        7 lag features
        5 rolling-sum features
        5 rolling-mean features

    Therefore:

        17 × 5 = 85

    Plus:

        minute-of-day sin/cos
        day-of-week sin/cos

    Therefore:

        85 + 4 = 89
    """

    features = []

    for source in SOURCE_SERIES:

        # ------------------------------
        # Lag features
        # ------------------------------

        for lag in LAGS:
            features.append(
                f"{source}_lag_{lag}"
            )

        # ------------------------------
        # Rolling sum features
        # ------------------------------

        for window in ROLLING_WINDOWS:
            features.append(
                f"{source}_rolling_sum_{window}"
            )

        # ------------------------------
        # Rolling mean features
        # ------------------------------

        for window in ROLLING_WINDOWS:
            features.append(
                f"{source}_rolling_mean_{window}"
            )

    # ------------------------------
    # Calendar features
    # ------------------------------

    features.extend([
        "minute_of_day_sin",
        "minute_of_day_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    ])

    return features


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(df):
    """
    Create chronological train / validation /
    calibration / locked-test partitions.

    Approximate proportions based on the structure
    of Paper 4's failure-risk experiment:

        Training       ≈ 69.4%
        Validation     ≈ 7.4%
        Calibration    ≈ 7.4%
        Locked test    ≈ 14.9%

    Because this project currently uses only BurstGPT_1,
    absolute Paper 4 minute boundaries cannot be reused.

    No random splitting is performed.
    """

    n = len(df)

    train_end = int(n * 0.694)

    validation_end = int(
        n * (0.694 + 0.074)
    )

    calibration_end = int(
        n * (0.694 + 0.074 + 0.074)
    )

    train = df.iloc[
        :train_end
    ].copy()

    validation = df.iloc[
        train_end:validation_end
    ].copy()

    calibration = df.iloc[
        validation_end:calibration_end
    ].copy()

    test = df.iloc[
        calibration_end:
    ].copy()

    return (
        train,
        validation,
        calibration,
        test,
    )


# ============================================================
# FEATURE MATRIX VALIDATION
# ============================================================

def validate_feature_matrix(
    df,
    feature_cols
):
    """
    Validate the usable feature matrix.

    This function is called AFTER removing the 1440-minute
    warm-up region, so complete causal features should contain
    no NaN values.
    """

    print("\n[4] Validating usable feature matrix...")

    # --------------------------------------------------------
    # Check all expected features exist
    # --------------------------------------------------------

    missing = [
        col
        for col in feature_cols
        if col not in df.columns
    ]

    if missing:

        print("\nMissing features:")

        for col in missing:
            print(f"  {col}")

        raise ValueError(
            "Expected Paper 4 risk features are missing."
        )

    # --------------------------------------------------------
    # Check feature count
    # --------------------------------------------------------

    actual_count = len(feature_cols)

    print(
        f"Expected features : "
        f"{EXPECTED_FEATURE_COUNT}"
    )

    print(
        f"Actual features   : "
        f"{actual_count}"
    )

    if actual_count != EXPECTED_FEATURE_COUNT:

        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} "
            f"features but constructed {actual_count}."
        )

    # --------------------------------------------------------
    # Check NaNs
    # --------------------------------------------------------

    nan_count = int(
        df[feature_cols]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Feature NaNs      : "
        f"{nan_count}"
    )

    if nan_count != 0:

        raise ValueError(
            f"Feature matrix contains "
            f"{nan_count} NaN values."
        )

    # --------------------------------------------------------
    # Check target
    # --------------------------------------------------------

    if "failure" not in df.columns:

        raise ValueError(
            "Target column 'failure' not found."
        )

    target_values = sorted(
        df["failure"]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        f"Target values     : "
        f"{target_values}"
    )

    if target_values != [0, 1]:

        raise ValueError(
            "Failure target must contain exactly "
            "[0, 1]."
        )

    # --------------------------------------------------------
    # Check minute column
    # --------------------------------------------------------

    if "minute" not in df.columns:

        raise ValueError(
            "Expected temporal column 'minute' "
            "was not found."
        )

    # --------------------------------------------------------
    # Check minute ordering
    # --------------------------------------------------------

    if not df["minute"].is_monotonic_increasing:

        raise ValueError(
            "Minute column is not monotonically increasing."
        )

    # --------------------------------------------------------
    # Check duplicate minutes
    # --------------------------------------------------------

    duplicate_minutes = int(
        df["minute"].duplicated().sum()
    )

    print(
        f"Duplicate minutes : "
        f"{duplicate_minutes}"
    )

    if duplicate_minutes != 0:

        raise ValueError(
            "Duplicate minute values detected."
        )

    print(
        "Feature matrix validation: PASS"
    )


# ============================================================
# FAILURE RATE
# ============================================================

def calculate_failure_rate(y):

    return float(
        y.mean()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "PAPER 4 XGBOOST FAILURE-RISK MODEL"
    )
    print("=" * 70)

    # ========================================================
    # 1. LOAD FEATURE MATRIX
    # ========================================================

    print(
        "\n[1] Loading failure-risk feature matrix..."
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nInput file not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows loaded: "
        f"{len(df):,}"
    )

    print(
        f"Columns: "
        f"{len(df.columns)}"
    )

    # ========================================================
    # 2. BUILD FEATURE LIST
    # ========================================================

    print(
        "\n[2] Building Paper 4 feature list..."
    )

    feature_cols = build_feature_names()

    print(
        f"Risk predictors: "
        f"{len(feature_cols)}"
    )

    if len(feature_cols) != EXPECTED_FEATURE_COUNT:

        raise RuntimeError(
            "Paper 4 feature construction failed."
        )

    # ========================================================
    # 3. REMOVE 1440-MINUTE WARM-UP
    # ========================================================

    print(
        "\n[3] Removing initial temporal warm-up rows..."
    )

    MAX_HISTORY = 1440

    if len(df) <= MAX_HISTORY:

        raise ValueError(
            "Dataset is too short for the "
            "1440-minute historical window."
        )

    usable = df.iloc[
        MAX_HISTORY:
    ].copy()

    usable = usable.reset_index(
        drop=True
    )

    print(
        f"Rows after warm-up: "
        f"{len(usable):,}"
    )

    print(
        f"First usable minute: "
        f"{usable['minute'].iloc[0]}"
    )

    print(
        f"Last usable minute: "
        f"{usable['minute'].iloc[-1]}"
    )

    # ========================================================
    # 4. VALIDATE
    # ========================================================

    validate_feature_matrix(
        usable,
        feature_cols
    )

    # ========================================================
    # 5. CHRONOLOGICAL PARTITIONS
    # ========================================================

    print(
        "\n[5] Creating chronological partitions..."
    )

    (
        train,
        validation,
        calibration,
        test,
    ) = chronological_split(
        usable
    )

    print(
        "\nSplit sizes:"
    )

    print(
        f"Training       : "
        f"{len(train):,}"
    )

    print(
        f"Validation     : "
        f"{len(validation):,}"
    )

    print(
        f"Calibration    : "
        f"{len(calibration):,}"
    )

    print(
        f"Locked test    : "
        f"{len(test):,}"
    )

    # ========================================================
    # 6. TEMPORAL ORDER CHECK
    # ========================================================

    print(
        "\n[6] Checking chronological ordering..."
    )

    train_min = int(
        train["minute"].min()
    )

    train_max = int(
        train["minute"].max()
    )

    validation_min = int(
        validation["minute"].min()
    )

    validation_max = int(
        validation["minute"].max()
    )

    calibration_min = int(
        calibration["minute"].min()
    )

    calibration_max = int(
        calibration["minute"].max()
    )

    test_min = int(
        test["minute"].min()
    )

    test_max = int(
        test["minute"].max()
    )

    print(
        f"Training       : "
        f"{train_min} -> {train_max}"
    )

    print(
        f"Validation     : "
        f"{validation_min} -> {validation_max}"
    )

    print(
        f"Calibration    : "
        f"{calibration_min} -> {calibration_max}"
    )

    print(
        f"Locked test    : "
        f"{test_min} -> {test_max}"
    )

    chronological_ok = (
        train_max < validation_min
        and
        validation_max < calibration_min
        and
        calibration_max < test_min
    )

    if not chronological_ok:

        raise RuntimeError(
            "Chronological partition validation failed."
        )

    print(
        "Chronological ordering: PASS"
    )

    # ========================================================
    # 7. PREPARE X / Y
    # ========================================================

    print(
        "\n[7] Preparing model matrices..."
    )

    X_train = train[
        feature_cols
    ]

    y_train = train[
        "failure"
    ].astype(int)

    X_validation = validation[
        feature_cols
    ]

    y_validation = validation[
        "failure"
    ].astype(int)

    X_calibration = calibration[
        feature_cols
    ]

    y_calibration = calibration[
        "failure"
    ].astype(int)

    X_test = test[
        feature_cols
    ]

    y_test = test[
        "failure"
    ].astype(int)

    # ========================================================
    # 8. FAILURE DISTRIBUTION
    # ========================================================

    print(
        "\n[8] Failure-rate distribution..."
    )

    train_failure_rate = calculate_failure_rate(
        y_train
    )

    validation_failure_rate = calculate_failure_rate(
        y_validation
    )

    calibration_failure_rate = calculate_failure_rate(
        y_calibration
    )

    test_failure_rate = calculate_failure_rate(
        y_test
    )

    print(
        f"Training failure rate    : "
        f"{train_failure_rate * 100:.4f}%"
    )

    print(
        f"Validation failure rate  : "
        f"{validation_failure_rate * 100:.4f}%"
    )

    print(
        f"Calibration failure rate : "
        f"{calibration_failure_rate * 100:.4f}%"
    )

    print(
        f"Test failure rate        : "
        f"{test_failure_rate * 100:.4f}%"
    )

    # ========================================================
    # 9. TRAIN XGBOOST
    # ========================================================

    print(
        "\n[9] Training Paper 4 XGBoost model..."
    )

    print(
        "\nHyperparameters:"
    )

    for key, value in XGB_PARAMS.items():

        print(
            f"  {key}: {value}"
        )

    model = XGBClassifier(
        **XGB_PARAMS
    )

    # IMPORTANT:
    # Only training data is used here.
    #
    # Validation, calibration and test remain untouched
    # during model fitting.

    model.fit(
        X_train,
        y_train
    )

    print(
        "\nModel training complete."
    )

    # ========================================================
    # 10. GENERATE PROBABILITIES
    # ========================================================

    print(
        "\n[10] Generating risk probabilities..."
    )

    train_probability = (
        model
        .predict_proba(
            X_train
        )[:, 1]
    )

    validation_probability = (
        model
        .predict_proba(
            X_validation
        )[:, 1]
    )

    calibration_probability = (
        model
        .predict_proba(
            X_calibration
        )[:, 1]
    )

    test_probability = (
        model
        .predict_proba(
            X_test
        )[:, 1]
    )

    # ========================================================
    # 11. PROBABILITY VALIDATION
    # ========================================================

    all_probabilities = np.concatenate([
        train_probability,
        validation_probability,
        calibration_probability,
        test_probability,
    ])

    probability_min = float(
        all_probabilities.min()
    )

    probability_max = float(
        all_probabilities.max()
    )

    print(
        f"Probability minimum: "
        f"{probability_min:.6f}"
    )

    print(
        f"Probability maximum: "
        f"{probability_max:.6f}"
    )

    if (
        probability_min < 0
        or probability_max > 1
    ):

        raise RuntimeError(
            "Predicted probabilities are outside [0, 1]."
        )

    print(
        "Probability validation: PASS"
    )

    # ========================================================
    # 12. SAVE MODEL
    # ========================================================

    print(
        "\n[11] Saving trained model..."
    )

    model_path = (
        MODEL_DIR
        /
        "paper4_xgboost_failure_risk.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Model saved:"
    )

    print(
        f"  {model_path}"
    )

    # ========================================================
    # 13. SAVE PREDICTIONS
    # ========================================================

    print(
        "\n[12] Saving risk predictions..."
    )

    prediction_frames = []

    split_data = [
        (
            "train",
            train,
            y_train,
            train_probability
        ),
        (
            "validation",
            validation,
            y_validation,
            validation_probability
        ),
        (
            "calibration",
            calibration,
            y_calibration,
            calibration_probability
        ),
        (
            "test",
            test,
            y_test,
            test_probability
        ),
    ]

    for (
        split_name,
        split_df,
        y,
        probability
    ) in split_data:

        temp = pd.DataFrame({
            "minute": split_df[
                "minute"
            ].values,

            "split": split_name,

            "failure": y.values,

            "predicted_failure_probability":
                probability,
        })

        prediction_frames.append(
            temp
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True
    )

    prediction_path = (
        RESULT_DIR
        /
        "failure_risk_predictions.csv"
    )

    predictions.to_csv(
        prediction_path,
        index=False
    )

    print(
        f"Predictions saved:"
    )

    print(
        f"  {prediction_path}"
    )

    # ========================================================
    # 14. SAVE METADATA
    # ========================================================

    print(
        "\n[13] Saving experiment metadata..."
    )

    metadata = {

        "model":
            "Paper 4 XGBoost failure-risk model",

        "dataset":
            "BurstGPT_1.csv",

        "feature_count":
            len(feature_cols),

        "features":
            feature_cols,

        "maximum_history_minutes":
            MAX_HISTORY,

        "warmup_rows_removed":
            MAX_HISTORY,

        "usable_rows":
            len(usable),

        "split_method":
            (
                "Chronological proportional split "
                "adapted to the available "
                "BurstGPT_1 trace"
            ),

        "split_proportions": {
            "training": 0.694,
            "validation": 0.074,
            "calibration": 0.074,
            "test": 0.149,
        },

        "split_sizes": {
            "train":
                len(train),

            "validation":
                len(validation),

            "calibration":
                len(calibration),

            "test":
                len(test),
        },

        "split_boundaries": {

            "train":
                {
                    "start":
                        train_min,

                    "end":
                        train_max,
                },

            "validation":
                {
                    "start":
                        validation_min,

                    "end":
                        validation_max,
                },

            "calibration":
                {
                    "start":
                        calibration_min,

                    "end":
                        calibration_max,
                },

            "test":
                {
                    "start":
                        test_min,

                    "end":
                        test_max,
                },
        },

        "failure_rates": {

            "train":
                train_failure_rate,

            "validation":
                validation_failure_rate,

            "calibration":
                calibration_failure_rate,

            "test":
                test_failure_rate,
        },

        "xgboost_parameters":
            XGB_PARAMS,

        "calibration":
            {
                "status":
                    "not performed",

                "module":
                    "08_calibration.py",
            },

        "test_status":
            (
                "Predictions generated. "
                "Test evaluation is performed "
                "in later modules."
            ),
    }

    metadata_path = (
        META_DIR
        /
        "failure_risk_training_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print(
        f"Metadata saved:"
    )

    print(
        f"  {metadata_path}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "FAILURE-RISK MODEL TRAINING COMPLETE"
    )

    print(
        "-" * 70
    )

    print(
        "\nCreated:"
    )

    print(
        "  models/failure_risk/"
        "paper4_xgboost_failure_risk.joblib"
    )

    print(
        "  results/predictions/"
        "failure_risk_predictions.csv"
    )

    print(
        "  results/validation/"
        "failure_risk_training_metadata.json"
    )

    print(
        "\nImportant:"
    )

    print(
        "  Calibration: NOT performed"
    )

    print(
        "  Test data: NOT used for training"
    )

    print(
        "  Threshold selection: NOT performed"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "07 COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()