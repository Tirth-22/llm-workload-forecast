from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier


# ======================================================================
# MODULE 16
# FAILURE-RISK FEATURE ABLATION
#
# Research question:
# Does removing historical zero-response features materially change
# failure-risk prediction performance?
#
# Full model:
#   89 Paper 4 risk features
#
# Ablation model:
#   72 features
#   17 historical zero-response features removed
#
# Everything else remains identical.
# ======================================================================


# ======================================================================
# PATHS
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "failure_features_minute.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "ablation"
)

VALIDATION_DIR = (
    BASE_DIR
    / "results"
    / "validation"
)


# ======================================================================
# EXPERIMENT CONFIGURATION
# ======================================================================

WARMUP_MINUTES = 1440

# Same chronological boundaries used by Module 07
TRAIN_END = 61395
VALIDATION_END = 67788
CALIBRATION_END = 74181

# Same fixed Paper 4 threshold used in Module 09 / 12
THETA_BASE = 0.9591

RANDOM_STATE = 42


# ======================================================================
# PAPER 4 XGBOOST CONFIGURATION
# Same configuration as Module 07
# ======================================================================

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
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ======================================================================
# PAPER 4 FEATURE DEFINITION
# ======================================================================

# Five source series from Paper 4.
SOURCE_PREFIXES = [
    "request_count",
    "completed_tokens",
    "gpt4_share",
    "api_share",
    "zero_response_count",
]


# Four calendar features.
CALENDAR_FEATURES = [
    "minute_of_day_sin",
    "minute_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]


# Expected historical feature structure:
#
# 7 lags:
#   1, 2, 5, 15, 60, 240, 1440
#
# 5 rolling sums:
#   5, 15, 60, 240, 1440
#
# 5 rolling means:
#   5, 15, 60, 240, 1440
#
# 17 features per source × 5 sources = 85
# 85 + 4 calendar = 89
#
# We validate the actual columns rather than assuming names blindly.


# ======================================================================
# HELPERS
# ======================================================================

def expected_calibration_error(
    y_true,
    probabilities,
    n_bins=10,
):
    """
    Calculate 10-bin Expected Calibration Error.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bins = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        if i == n_bins - 1:

            mask = (
                (probabilities >= bins[i])
                &
                (probabilities <= bins[i + 1])
            )

        else:

            mask = (
                (probabilities >= bins[i])
                &
                (probabilities < bins[i + 1])
            )

        if not np.any(mask):
            continue

        bin_accuracy = np.mean(
            y_true[mask]
        )

        bin_confidence = np.mean(
            probabilities[mask]
        )

        bin_fraction = np.mean(mask)

        ece += (
            bin_fraction
            * abs(
                bin_accuracy
                - bin_confidence
            )
        )

    return float(ece)


def calculate_metrics(
    y_true,
    raw_probability,
    calibrated_probability,
    threshold,
):
    """
    Calculate the same core risk metrics used in Module 12.
    """

    y_true = np.asarray(y_true)

    raw_probability = np.asarray(
        raw_probability
    )

    calibrated_probability = np.asarray(
        calibrated_probability
    )

    predictions = (
        calibrated_probability
        >= threshold
    ).astype(int)

    return {
        "test_rows":
            int(len(y_true)),

        "failure_prevalence_percent":
            float(
                np.mean(y_true) * 100.0
            ),

        "ROC_AUC":
            float(
                roc_auc_score(
                    y_true,
                    calibrated_probability
                )
            ),

        "Average_Precision":
            float(
                average_precision_score(
                    y_true,
                    calibrated_probability
                )
            ),

        "Brier_calibrated":
            float(
                brier_score_loss(
                    y_true,
                    calibrated_probability
                )
            ),

        "Brier_raw_diagnostic":
            float(
                brier_score_loss(
                    y_true,
                    raw_probability
                )
            ),

        "LogLoss_calibrated":
            float(
                log_loss(
                    y_true,
                    np.clip(
                        calibrated_probability,
                        1e-15,
                        1.0 - 1e-15
                    )
                )
            ),

        "ECE_10_bin":
            expected_calibration_error(
                y_true,
                calibrated_probability,
                n_bins=10
            ),

        "classification_threshold":
            float(threshold),

        "Precision":
            float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0
                )
            ),

        "Recall":
            float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0
                )
            ),

        "F1":
            float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0
                )
            ),
    }


def save_json(
    data,
    path,
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2
        )


def build_paper4_feature_list(columns):
    """
    Construct the exact 89 Paper 4 risk features.

    We select:
      - 17 historical features for each of 5 source series
      - 4 calendar features

    The function preserves the original CSV column order.
    """

    columns = list(columns)

    # --------------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------------

    selected = []

    for column in columns:

        if column in CALENDAR_FEATURES:

            selected.append(column)

    # --------------------------------------------------------------
    # Historical source features
    # --------------------------------------------------------------

    for column in columns:

        if column in CALENDAR_FEATURES:
            continue

        for prefix in SOURCE_PREFIXES:

            if column.startswith(
                prefix + "_"
            ):

                selected.append(
                    column
                )

                break

    # Remove accidental duplicates while preserving order.
    selected = list(
        dict.fromkeys(selected)
    )

    return selected


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("MODULE 16 — FAILURE-RISK FEATURE ABLATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==================================================================
    # LOAD DATA
    # ==================================================================

    print(
        "\nLoading failure-risk feature dataset..."
    )

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Required input not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # ==================================================================
    # REQUIRED SCHEMA
    # ==================================================================

    required_columns = {
        "minute",
        "failure",
    }

    missing_required = (
        required_columns
        - set(df.columns)
    )

    if missing_required:

        raise ValueError(
            "Missing required columns: "
            + str(
                sorted(
                    missing_required
                )
            )
        )

    print(
        "Required schema: PASS"
    )

    # ==================================================================
    # SORT + VALIDATE
    # ==================================================================

    df = (
        df
        .sort_values("minute")
        .reset_index(drop=True)
    )

    if df["minute"].duplicated().any():

        raise ValueError(
            "Duplicate minute indices detected."
        )

    if not set(
        df["failure"].unique()
    ).issubset({0, 1}):

        raise ValueError(
            "Failure target must contain only 0 and 1."
        )

    print(
        "Temporal ordering: PASS"
    )

    # ==================================================================
    # REMOVE WARM-UP
    # ==================================================================

    df_model = df[
        df["minute"] >= WARMUP_MINUTES
    ].copy()

    print(
        f"Warm-up removed: "
        f"{WARMUP_MINUTES} minutes"
    )

    print(
        f"Usable rows: "
        f"{len(df_model):,}"
    )

    # ==================================================================
    # IDENTIFY EXACT PAPER 4 FEATURES
    # ==================================================================

    risk_features = build_paper4_feature_list(
        df_model.columns
    )

    print(
        "\nDetected Paper 4 risk features: "
        f"{len(risk_features)}"
    )

    # Print them for validation.
    print(
        "\nDetected feature list:"
    )

    for i, feature in enumerate(
        risk_features,
        start=1
    ):

        print(
            f"{i:02d}. {feature}"
        )

    # ==================================================================
    # EXACT 89-FEATURE VALIDATION
    # ==================================================================

    if len(risk_features) != 89:

        raise ValueError(
            "\nExpected exactly 89 Paper 4 "
            "risk features, but found "
            f"{len(risk_features)}.\n\n"
            "The detected features were:\n"
            + "\n".join(
                risk_features
            )
        )

    print(
        "\nPaper 4 89-feature definition: PASS"
    )

    # ==================================================================
    # IDENTIFY ZERO-RESPONSE FEATURES
    # ==================================================================

    zero_response_features = [
        column
        for column in risk_features
        if column.startswith(
            "zero_response_count_"
        )
    ]

    print(
        "\nZero-response historical features: "
        f"{len(zero_response_features)}"
    )

    print(
        "\nFeatures removed for ablation:"
    )

    for feature in zero_response_features:

        print(
            f"  - {feature}"
        )

    # ==================================================================
    # VALIDATE 17 ZERO-RESPONSE FEATURES
    # ==================================================================

    if len(zero_response_features) != 17:

        raise ValueError(
            "Expected 17 historical "
            "zero-response features, "
            f"found {len(zero_response_features)}."
        )

    # ==================================================================
    # BUILD ABLATION FEATURE LIST
    # ==================================================================

    ablation_features = [
        column
        for column in risk_features
        if column
        not in zero_response_features
    ]

    print(
        "\nFull model features: "
        f"{len(risk_features)}"
    )

    print(
        "Ablation model features: "
        f"{len(ablation_features)}"
    )

    if len(ablation_features) != 72:

        raise ValueError(
            "Expected 72 ablation features, "
            f"found {len(ablation_features)}."
        )

    print(
        "Feature ablation definition: PASS"
    )

    # ==================================================================
    # BUILD MATRICES
    # ==================================================================

    y = (
        df_model["failure"]
        .astype(int)
    )

    X_full = df_model[
        risk_features
    ].copy()

    X_ablation = df_model[
        ablation_features
    ].copy()

    # ==================================================================
    # NaN VALIDATION
    # ==================================================================

    full_nan_count = int(
        X_full.isna().sum().sum()
    )

    ablation_nan_count = int(
        X_ablation.isna().sum().sum()
    )

    print(
        "\nFull feature NaNs: "
        f"{full_nan_count}"
    )

    print(
        "Ablation feature NaNs: "
        f"{ablation_nan_count}"
    )

    if full_nan_count != 0:

        raise ValueError(
            "Full feature matrix contains NaNs."
        )

    if ablation_nan_count != 0:

        raise ValueError(
            "Ablation feature matrix contains NaNs."
        )

    print(
        "Feature NaN validation: PASS"
    )

    # ==================================================================
    # TEMPORAL SPLITS
    # ==================================================================

    train_mask = (
        df_model["minute"]
        <= TRAIN_END
    )

    validation_mask = (
        (df_model["minute"] > TRAIN_END)
        &
        (df_model["minute"] <= VALIDATION_END)
    )

    calibration_mask = (
        (df_model["minute"] > VALIDATION_END)
        &
        (df_model["minute"] <= CALIBRATION_END)
    )

    test_mask = (
        df_model["minute"]
        > CALIBRATION_END
    )

    masks = {
        "train": train_mask,
        "validation": validation_mask,
        "calibration": calibration_mask,
        "test": test_mask,
    }

    print(
        "\nChronological split:"
    )

    for name, mask in masks.items():

        subset = df_model.loc[
            mask
        ]

        if len(subset) == 0:

            raise ValueError(
                f"{name} split is empty."
            )

        print(
            f"{name.capitalize():12s}: "
            f"{len(subset):,} rows | "
            f"{subset['minute'].min()} -> "
            f"{subset['minute'].max()}"
        )

    print(
        "Chronological split validation: PASS"
    )

    # ==================================================================
    # TRAIN FULL MODEL
    # ==================================================================

    print(
        "\nTraining FULL 89-feature XGBoost model..."
    )

    full_model = XGBClassifier(
        **XGB_PARAMS
    )

    full_model.fit(
        X_full.loc[train_mask],
        y.loc[train_mask]
    )

    print(
        "Full model training complete."
    )

    # ==================================================================
    # TRAIN ABLATION MODEL
    # ==================================================================

    print(
        "\nTraining ABLATION 72-feature XGBoost model..."
    )

    ablation_model = XGBClassifier(
        **XGB_PARAMS
    )

    ablation_model.fit(
        X_ablation.loc[train_mask],
        y.loc[train_mask]
    )

    print(
        "Ablation model training complete."
    )

    # ==================================================================
    # CALIBRATION DATA
    # ==================================================================

    y_calibration = y.loc[
        calibration_mask
    ].to_numpy()

    full_calibration_raw = (
        full_model.predict_proba(
            X_full.loc[
                calibration_mask
            ]
        )[:, 1]
    )

    ablation_calibration_raw = (
        ablation_model.predict_proba(
            X_ablation.loc[
                calibration_mask
            ]
        )[:, 1]
    )

    print(
        "\nCalibration rows: "
        f"{len(y_calibration):,}"
    )

    # ==================================================================
    # FIT ISOTONIC CALIBRATORS
    # ==================================================================

    print(
        "\nFitting FULL isotonic calibrator..."
    )

    full_calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip"
    )

    full_calibrator.fit(
        full_calibration_raw,
        y_calibration
    )

    print(
        "Full calibration complete."
    )

    print(
        "\nFitting ABLATION isotonic calibrator..."
    )

    ablation_calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip"
    )

    ablation_calibrator.fit(
        ablation_calibration_raw,
        y_calibration
    )

    print(
        "Ablation calibration complete."
    )

    # ==================================================================
    # CALIBRATION DIAGNOSTICS
    # ==================================================================

    full_calibration_calibrated = (
        full_calibrator.predict(
            full_calibration_raw
        )
    )

    ablation_calibration_calibrated = (
        ablation_calibrator.predict(
            ablation_calibration_raw
        )
    )

    full_calibration_brier = (
        brier_score_loss(
            y_calibration,
            full_calibration_calibrated
        )
    )

    ablation_calibration_brier = (
        brier_score_loss(
            y_calibration,
            ablation_calibration_calibrated
        )
    )

    print(
        "\nCalibration Brier scores:"
    )

    print(
        f"Full model:     "
        f"{full_calibration_brier:.6f}"
    )

    print(
        f"Ablation model: "
        f"{ablation_calibration_brier:.6f}"
    )

    # ==================================================================
    # LOCKED TEST PREDICTIONS
    # ==================================================================

    print(
        "\nGenerating locked-test predictions..."
    )

    y_test = y.loc[
        test_mask
    ].to_numpy()

    full_test_raw = (
        full_model.predict_proba(
            X_full.loc[
                test_mask
            ]
        )[:, 1]
    )

    ablation_test_raw = (
        ablation_model.predict_proba(
            X_ablation.loc[
                test_mask
            ]
        )[:, 1]
    )

    full_test_calibrated = (
        full_calibrator.predict(
            full_test_raw
        )
    )

    ablation_test_calibrated = (
        ablation_calibrator.predict(
            ablation_test_raw
        )
    )

    print(
        f"Locked-test rows: "
        f"{len(y_test):,}"
    )

    # ==================================================================
    # TEST METRICS
    # ==================================================================

    full_metrics = calculate_metrics(
        y_true=y_test,
        raw_probability=full_test_raw,
        calibrated_probability=full_test_calibrated,
        threshold=THETA_BASE,
    )

    ablation_metrics = calculate_metrics(
        y_true=y_test,
        raw_probability=ablation_test_raw,
        calibrated_probability=ablation_test_calibrated,
        threshold=THETA_BASE,
    )

    # ==================================================================
    # DISPLAY RESULTS
    # ==================================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOCKED-TEST ABLATION RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        "\nFULL MODEL — 89 FEATURES"
    )

    for key, value in full_metrics.items():

        if isinstance(value, float):

            print(
                f"{key:35s}: "
                f"{value:.6f}"
            )

        else:

            print(
                f"{key:35s}: "
                f"{value}"
            )

    print(
        "\nABLATION MODEL — 72 FEATURES"
    )

    for key, value in ablation_metrics.items():

        if isinstance(value, float):

            print(
                f"{key:35s}: "
                f"{value:.6f}"
            )

        else:

            print(
                f"{key:35s}: "
                f"{value}"
            )

    # ==================================================================
    # METRIC COMPARISON
    # ==================================================================

    comparison_rows = []

    metric_names = [
        "ROC_AUC",
        "Average_Precision",
        "Brier_calibrated",
        "Brier_raw_diagnostic",
        "LogLoss_calibrated",
        "ECE_10_bin",
        "Precision",
        "Recall",
        "F1",
    ]

    for metric in metric_names:

        full_value = full_metrics[
            metric
        ]

        ablation_value = ablation_metrics[
            metric
        ]

        comparison_rows.append(
            {
                "metric":
                    metric,

                "full_89_features":
                    full_value,

                "ablation_72_features":
                    ablation_value,

                "difference_ablation_minus_full":
                    ablation_value
                    - full_value,
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_path = (
        OUTPUT_DIR
        / "risk_feature_ablation_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    print(
        "\nCreated:"
    )

    print(
        "  results/ablation/"
        "risk_feature_ablation_comparison.csv"
    )

    # ==================================================================
    # SAVE TEST PREDICTIONS
    # ==================================================================

    prediction_df = pd.DataFrame(
        {
            "minute":
                df_model.loc[
                    test_mask,
                    "minute"
                ].to_numpy(),

            "failure":
                y_test,

            "full_raw_probability":
                full_test_raw,

            "full_calibrated_probability":
                full_test_calibrated,

            "ablation_raw_probability":
                ablation_test_raw,

            "ablation_calibrated_probability":
                ablation_test_calibrated,

            "full_trigger":
                (
                    full_test_calibrated
                    >= THETA_BASE
                ).astype(int),

            "ablation_trigger":
                (
                    ablation_test_calibrated
                    >= THETA_BASE
                ).astype(int),
        }
    )

    prediction_path = (
        OUTPUT_DIR
        / "risk_feature_ablation_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    print(
        "  results/ablation/"
        "risk_feature_ablation_predictions.csv"
    )

    # ==================================================================
    # SAVE METADATA
    # ==================================================================

    metadata = {

        "module":
            "16_ablation_risk_features",

        "experiment":
            "Remove historical zero-response features",

        "full_feature_count":
            len(risk_features),

        "ablation_feature_count":
            len(ablation_features),

        "removed_feature_count":
            len(zero_response_features),

        "removed_features":
            zero_response_features,

        "warmup_minutes":
            WARMUP_MINUTES,

        "train_end":
            TRAIN_END,

        "validation_end":
            VALIDATION_END,

        "calibration_end":
            CALIBRATION_END,

        "theta_base":
            THETA_BASE,

        "xgboost_parameters":
            XGB_PARAMS,

        "test_used_for_training":
            False,

        "test_used_for_calibration":
            False,

        "test_used_for_threshold_selection":
            False,

        "models_retrained":
            True,

        "research_question":
            "Does removing historical "
            "zero-response features materially "
            "change failure-risk prediction "
            "performance?"
    }

    metadata_path = (
        VALIDATION_DIR
        / "risk_feature_ablation_metadata.json"
    )

    save_json(
        metadata,
        metadata_path
    )

    print(
        "  results/validation/"
        "risk_feature_ablation_metadata.json"
    )

    # ==================================================================
    # SAVE SUMMARY
    # ==================================================================

    summary = {

        "full_model":
            full_metrics,

        "ablation_model":
            ablation_metrics,

        "full_feature_count":
            len(risk_features),

        "ablation_feature_count":
            len(ablation_features),

        "removed_features":
            zero_response_features,
    }

    summary_path = (
        OUTPUT_DIR
        / "risk_feature_ablation_summary.json"
    )

    save_json(
        summary,
        summary_path
    )

    print(
        "  results/ablation/"
        "risk_feature_ablation_summary.json"
    )

    # ==================================================================
    # FINAL ALIGNMENT VALIDATION
    # ==================================================================

    if not (
        len(full_test_raw)
        == len(ablation_test_raw)
        == len(y_test)
    ):

        raise ValueError(
            "Full model, ablation model, "
            "and test target lengths do not match."
        )

    print(
        "\nAblation alignment validation: PASS"
    )

    # ==================================================================
    # FINAL STATUS
    # ==================================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODULE 16 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nFull model: "
        "89 Paper 4 risk features"
    )

    print(
        "Ablation model: "
        "72 features"
    )

    print(
        "Removed: "
        "17 historical zero-response features"
    )

    print(
        "\nNo test parameter was tuned."
    )

    print(
        "The locked test set was not used "
        "for training or calibration."
    )

    print(
        "No model was declared superior automatically."
    )


if __name__ == "__main__":
    main()