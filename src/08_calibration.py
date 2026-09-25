import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PREDICTIONS_PATH = (
    BASE_DIR
    / "results"
    / "predictions"
    / "failure_risk_predictions.csv"
)

CALIBRATION_DIR = (
    BASE_DIR
    / "models"
    / "calibration"
)

CALIBRATION_MODEL_PATH = (
    CALIBRATION_DIR
    / "isotonic_failure_calibrator.joblib"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "predictions"
)

CALIBRATED_PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "calibrated_failure_risk_predictions.csv"
)

VALIDATION_DIR = (
    BASE_DIR
    / "results"
    / "validation"
)

METADATA_PATH = (
    VALIDATION_DIR
    / "calibration_metadata.json"
)


# ======================================================================
# EXPECTED COLUMNS FROM MODULE 07
# ======================================================================

MINUTE_COLUMN = "minute"
SPLIT_COLUMN = "split"
TARGET_COLUMN = "failure"
RAW_PROBABILITY_COLUMN = "predicted_failure_probability"


# ======================================================================
# HELPER
# ======================================================================

def evaluate_probabilities(y_true, probabilities):
    """
    Calculate probability-quality metrics.
    """

    y_true = np.asarray(y_true, dtype=int)

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    probabilities = np.clip(
        probabilities,
        1e-7,
        1.0 - 1e-7
    )

    return {
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities
            )
        ),
        "mean_probability": float(
            np.mean(probabilities)
        ),
        "failure_rate": float(
            np.mean(y_true)
        )
    }


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("PAPER 4 PROBABILITY CALIBRATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. LOAD MODULE 07 PREDICTIONS
    # ------------------------------------------------------------------

    print("\n[1] Loading failure-risk predictions...")

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTIONS_PATH}"
        )

    df = pd.read_csv(
        PREDICTIONS_PATH
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    # ------------------------------------------------------------------
    # 2. VALIDATE MODULE 07 SCHEMA
    # ------------------------------------------------------------------

    print("\n[2] Validating Module 07 prediction schema...")

    required_columns = [
        MINUTE_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
        RAW_PROBABILITY_COLUMN
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required Module 07 columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    print(
        f"Minute column       : {MINUTE_COLUMN}"
    )

    print(
        f"Split column        : {SPLIT_COLUMN}"
    )

    print(
        f"Target column       : {TARGET_COLUMN}"
    )

    print(
        f"Raw probability     : {RAW_PROBABILITY_COLUMN}"
    )

    print("Module 07 schema validation: PASS")

    # ------------------------------------------------------------------
    # 3. BASIC DATA VALIDATION
    # ------------------------------------------------------------------

    print("\n[3] Validating prediction data...")

    required = [
        MINUTE_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
        RAW_PROBABILITY_COLUMN
    ]

    missing_values = (
        df[required]
        .isna()
        .sum()
    )

    if missing_values.any():
        print("\nMissing values:")

        print(
            missing_values[
                missing_values > 0
            ]
        )

        raise ValueError(
            "Prediction data contains missing values."
        )

    # Sort chronologically
    df = (
        df
        .sort_values(MINUTE_COLUMN)
        .reset_index(drop=True)
    )

    # Duplicate check
    duplicate_minutes = (
        df[MINUTE_COLUMN]
        .duplicated()
        .sum()
    )

    if duplicate_minutes > 0:
        raise ValueError(
            f"Duplicate minutes detected: "
            f"{duplicate_minutes}"
        )

    # Target validation
    target_values = sorted(
        df[TARGET_COLUMN]
        .astype(int)
        .unique()
        .tolist()
    )

    if target_values != [0, 1]:
        raise ValueError(
            f"Unexpected failure target values: "
            f"{target_values}"
        )

    # Probability validation
    probabilities = (
        df[RAW_PROBABILITY_COLUMN]
        .astype(float)
    )

    if (
        (probabilities < 0).any()
        or
        (probabilities > 1).any()
    ):
        raise ValueError(
            "Raw probabilities must be between 0 and 1."
        )

    print(
        "Missing values       : 0"
    )

    print(
        "Duplicate minutes    : 0"
    )

    print(
        f"Target values        : {target_values}"
    )

    print(
        "Probability range    : "
        f"{probabilities.min():.6f} -> "
        f"{probabilities.max():.6f}"
    )

    print(
        "Prediction validation: PASS"
    )

    # ------------------------------------------------------------------
    # 4. INSPECT MODULE 07 SPLITS
    # ------------------------------------------------------------------

    print("\n[4] Inspecting chronological partitions...")

    df[SPLIT_COLUMN] = (
        df[SPLIT_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    split_values = sorted(
        df[SPLIT_COLUMN]
        .unique()
        .tolist()
    )

    print(
        f"Available splits: {split_values}"
    )

    expected_splits = {
        "train",
        "validation",
        "calibration",
        "test"
    }

    missing_splits = (
        expected_splits
        - set(split_values)
    )

    if missing_splits:
        raise ValueError(
            "Required chronological split(s) missing: "
            f"{sorted(missing_splits)}"
        )

    for split_name in [
        "train",
        "validation",
        "calibration",
        "test"
    ]:

        split_df = df[
            df[SPLIT_COLUMN] == split_name
        ]

        print(
            f"{split_name.capitalize():<14}: "
            f"{len(split_df):,} rows | "
            f"{split_df[MINUTE_COLUMN].min()} -> "
            f"{split_df[MINUTE_COLUMN].max()}"
        )

    # ------------------------------------------------------------------
    # 5. EXTRACT CALIBRATION AND LOCKED TEST
    # ------------------------------------------------------------------

    print("\n[5] Extracting calibration and locked-test partitions...")

    calibration_mask = (
        df[SPLIT_COLUMN]
        == "calibration"
    )

    test_mask = (
        df[SPLIT_COLUMN]
        == "test"
    )

    calibration_df = (
        df.loc[
            calibration_mask
        ]
        .copy()
    )

    test_df = (
        df.loc[
            test_mask
        ]
        .copy()
    )

    if len(calibration_df) == 0:
        raise ValueError(
            "Calibration partition is empty."
        )

    if len(test_df) == 0:
        raise ValueError(
            "Locked test partition is empty."
        )

    print(
        f"Calibration rows : "
        f"{len(calibration_df):,}"
    )

    print(
        f"Test rows        : "
        f"{len(test_df):,}"
    )

    print(
        f"Calibration range: "
        f"{calibration_df[MINUTE_COLUMN].min()} -> "
        f"{calibration_df[MINUTE_COLUMN].max()}"
    )

    print(
        f"Test range       : "
        f"{test_df[MINUTE_COLUMN].min()} -> "
        f"{test_df[MINUTE_COLUMN].max()}"
    )

    # ------------------------------------------------------------------
    # 6. VALIDATE CALIBRATION PARTITION
    # ------------------------------------------------------------------

    print("\n[6] Validating calibration partition...")

    calibration_target = (
        calibration_df[TARGET_COLUMN]
        .astype(int)
        .to_numpy()
    )

    calibration_probability = (
        calibration_df[RAW_PROBABILITY_COLUMN]
        .astype(float)
        .to_numpy()
    )

    calibration_classes = np.unique(
        calibration_target
    )

    if len(calibration_classes) != 2:
        raise ValueError(
            "Calibration partition does not contain both classes."
        )

    print(
        f"Calibration failure rate: "
        f"{calibration_target.mean() * 100:.4f}%"
    )

    print(
        f"Calibration classes: "
        f"{calibration_classes.tolist()}"
    )

    print(
        "Calibration partition validation: PASS"
    )

    # ------------------------------------------------------------------
    # 7. FIT ISOTONIC CALIBRATION
    # ------------------------------------------------------------------

    print("\n[7] Fitting isotonic probability calibrator...")

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip"
    )

    calibrator.fit(
        calibration_probability,
        calibration_target
    )

    print(
        "Isotonic calibration fitted."
    )

    print(
        "Calibration source: "
        "CALIBRATION partition only"
    )

    # ------------------------------------------------------------------
    # 8. GENERATE CALIBRATED PROBABILITIES
    # ------------------------------------------------------------------

    print("\n[8] Generating calibrated probabilities...")

    all_raw_probabilities = (
        df[RAW_PROBABILITY_COLUMN]
        .astype(float)
        .to_numpy()
    )

    calibrated_probabilities = (
        calibrator.predict(
            all_raw_probabilities
        )
    )

    calibrated_probabilities = np.clip(
        calibrated_probabilities,
        0.0,
        1.0
    )

    df[
        "calibrated_failure_probability"
    ] = calibrated_probabilities

    print(
        "Raw probability range:"
    )

    print(
        f"  {all_raw_probabilities.min():.6f}"
        f" -> "
        f"{all_raw_probabilities.max():.6f}"
    )

    print(
        "Calibrated probability range:"
    )

    print(
        f"  {calibrated_probabilities.min():.6f}"
        f" -> "
        f"{calibrated_probabilities.max():.6f}"
    )

    print(
        "Calibrated probability validation: PASS"
    )

    # ------------------------------------------------------------------
    # 9. CALIBRATION-PARTITION EVALUATION
    # ------------------------------------------------------------------

    print("\n[9] Evaluating calibration partition...")

    calibration_calibrated_probability = (
        df.loc[
            calibration_mask,
            "calibrated_failure_probability"
        ]
        .astype(float)
        .to_numpy()
    )

    raw_calibration_metrics = (
        evaluate_probabilities(
            calibration_target,
            calibration_probability
        )
    )

    calibrated_calibration_metrics = (
        evaluate_probabilities(
            calibration_target,
            calibration_calibrated_probability
        )
    )

    print("\nRaw calibration probabilities:")

    print(
        f"  Brier score : "
        f"{raw_calibration_metrics['brier_score']:.6f}"
    )

    print(
        f"  Log loss    : "
        f"{raw_calibration_metrics['log_loss']:.6f}"
    )

    print("\nCalibrated probabilities:")

    print(
        f"  Brier score : "
        f"{calibrated_calibration_metrics['brier_score']:.6f}"
    )

    print(
        f"  Log loss    : "
        f"{calibrated_calibration_metrics['log_loss']:.6f}"
    )

    # ------------------------------------------------------------------
    # 10. LOCKED TEST DIAGNOSTIC EVALUATION
    # ------------------------------------------------------------------

    print("\n[10] Evaluating locked test diagnostically...")

    test_target = (
        test_df[TARGET_COLUMN]
        .astype(int)
        .to_numpy()
    )

    test_raw_probability = (
        test_df[RAW_PROBABILITY_COLUMN]
        .astype(float)
        .to_numpy()
    )

    test_calibrated_probability = (
        df.loc[
            test_mask,
            "calibrated_failure_probability"
        ]
        .astype(float)
        .to_numpy()
    )

    raw_test_metrics = (
        evaluate_probabilities(
            test_target,
            test_raw_probability
        )
    )

    calibrated_test_metrics = (
        evaluate_probabilities(
            test_target,
            test_calibrated_probability
        )
    )

    print("\nRaw locked-test probabilities:")

    print(
        f"  Brier score : "
        f"{raw_test_metrics['brier_score']:.6f}"
    )

    print(
        f"  Log loss    : "
        f"{raw_test_metrics['log_loss']:.6f}"
    )

    print("\nCalibrated locked-test probabilities:")

    print(
        f"  Brier score : "
        f"{calibrated_test_metrics['brier_score']:.6f}"
    )

    print(
        f"  Log loss    : "
        f"{calibrated_test_metrics['log_loss']:.6f}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  Test data was NOT used to fit calibration."
    )

    print(
        "  Test metrics are diagnostic only."
    )

    print(
        "  No risk threshold was selected."
    )

    # ------------------------------------------------------------------
    # 11. SAVE CALIBRATOR
    # ------------------------------------------------------------------

    print("\n[11] Saving calibration model...")

    CALIBRATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        calibrator,
        CALIBRATION_MODEL_PATH
    )

    print(
        "Calibration model saved:"
    )

    print(
        f"  {CALIBRATION_MODEL_PATH}"
    )

    # ------------------------------------------------------------------
    # 12. SAVE CALIBRATED PREDICTIONS
    # ------------------------------------------------------------------

    print("\n[12] Saving calibrated predictions...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        CALIBRATED_PREDICTIONS_PATH,
        index=False
    )

    print(
        "Calibrated predictions saved:"
    )

    print(
        f"  {CALIBRATED_PREDICTIONS_PATH}"
    )

    # ------------------------------------------------------------------
    # 13. SAVE METADATA
    # ------------------------------------------------------------------

    print("\n[13] Saving experiment metadata...")

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = {

        "module":
            "08_probability_calibration",

        "method":
            "isotonic_regression",

        "input_predictions":
            str(
                PREDICTIONS_PATH.relative_to(
                    BASE_DIR
                )
            ),

        "calibration_model":
            str(
                CALIBRATION_MODEL_PATH.relative_to(
                    BASE_DIR
                )
            ),

        "output_predictions":
            str(
                CALIBRATED_PREDICTIONS_PATH.relative_to(
                    BASE_DIR
                )
            ),

        "partition_source":
            "Module 07 split column",

        "calibration_split":
            "calibration",

        "test_split":
            "test",

        "calibration_start_minute":
            int(
                calibration_df[
                    MINUTE_COLUMN
                ].min()
            ),

        "calibration_end_minute":
            int(
                calibration_df[
                    MINUTE_COLUMN
                ].max()
            ),

        "calibration_rows":
            int(
                len(calibration_df)
            ),

        "test_start_minute":
            int(
                test_df[
                    MINUTE_COLUMN
                ].min()
            ),

        "test_end_minute":
            int(
                test_df[
                    MINUTE_COLUMN
                ].max()
            ),

        "test_rows":
            int(
                len(test_df)
            ),

        "calibration_failure_rate":
            float(
                calibration_target.mean()
            ),

        "raw_calibration_metrics":
            raw_calibration_metrics,

        "calibrated_calibration_metrics":
            calibrated_calibration_metrics,

        "raw_test_metrics":
            raw_test_metrics,

        "calibrated_test_metrics":
            calibrated_test_metrics,

        "test_used_for_fitting":
            False,

        "threshold_selection_performed":
            False,

        "capacity_simulation_performed":
            False,

        "calibration_validation":
            "PASS"
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    print(
        "Metadata saved:"
    )

    print(
        f"  {METADATA_PATH}"
    )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PROBABILITY CALIBRATION COMPLETE")
    print("=" * 70)

    print("\nCreated:")

    print(
        "  models/calibration/"
        "isotonic_failure_calibrator.joblib"
    )

    print(
        "  results/predictions/"
        "calibrated_failure_risk_predictions.csv"
    )

    print(
        "  results/validation/"
        "calibration_metadata.json"
    )

    print("\nImportant:")

    print(
        "  Calibration fitted on calibration partition only"
    )

    print(
        "  Locked test was NOT used for fitting"
    )

    print(
        "  Risk threshold NOT selected"
    )

    print(
        "  Capacity simulation NOT performed"
    )

    print(
        "\n08 COMPLETE"
    )


if __name__ == "__main__":
    main()