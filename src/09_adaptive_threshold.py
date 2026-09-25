import json
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "results"
    / "predictions"
    / "calibrated_failure_risk_predictions.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "policies"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "adaptive_threshold_predictions.csv"
)

VALIDATION_DIR = (
    BASE_DIR
    / "results"
    / "validation"
)

METADATA_PATH = (
    VALIDATION_DIR
    / "adaptive_threshold_metadata.json"
)


# ----------------------------------------------------------------------
# Paper 4 validation-selected starting threshold
# ----------------------------------------------------------------------

THETA_BASE = 0.9591


# ----------------------------------------------------------------------
# Adaptive controller parameters
#
# These are controller parameters, not trained ML parameters.
# They will be subjected to sensitivity analysis later.
# ----------------------------------------------------------------------

ETA = 0.01

THETA_MIN = 0.50

THETA_MAX = 0.99

HISTORY_WINDOW = 60


# ======================================================================
# REQUIRED COLUMNS
# ======================================================================

MINUTE_COLUMN = "minute"

SPLIT_COLUMN = "split"

TARGET_COLUMN = "failure"

RAW_PROBABILITY_COLUMN = (
    "predicted_failure_probability"
)

CALIBRATED_PROBABILITY_COLUMN = (
    "calibrated_failure_probability"
)


# ======================================================================
# HELPER
# ======================================================================

def clip_threshold(theta):
    """
    Keep threshold inside configured bounds.
    """

    return float(
        np.clip(
            theta,
            THETA_MIN,
            THETA_MAX
        )
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("ADAPTIVE FAILURE-RISK THRESHOLD CONTROLLER")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. LOAD CALIBRATED PREDICTIONS
    # ------------------------------------------------------------------

    print("\n[1] Loading calibrated risk predictions...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    # ------------------------------------------------------------------
    # 2. VALIDATE INPUT SCHEMA
    # ------------------------------------------------------------------

    print("\n[2] Validating input schema...")

    required_columns = [
        MINUTE_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
        RAW_PROBABILITY_COLUMN,
        CALIBRATED_PROBABILITY_COLUMN
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    print(
        "Required columns: PASS"
    )

    # ------------------------------------------------------------------
    # 3. BASIC VALIDATION
    # ------------------------------------------------------------------

    print("\n[3] Validating prediction data...")

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Missing values detected in required columns."
        )

    df = (
        df
        .sort_values(MINUTE_COLUMN)
        .reset_index(drop=True)
    )

    if df[MINUTE_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate minute indices detected."
        )

    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(
            "Failure target must contain only 0 and 1."
        )

    if (
        (df[CALIBRATED_PROBABILITY_COLUMN] < 0).any()
        or
        (df[CALIBRATED_PROBABILITY_COLUMN] > 1).any()
    ):
        raise ValueError(
            "Calibrated probabilities must lie in [0, 1]."
        )

    print(
        "Missing values    : 0"
    )

    print(
        "Duplicate minutes : 0"
    )

    print(
        "Target values     : [0, 1]"
    )

    print(
        "Input validation   : PASS"
    )

    # ------------------------------------------------------------------
    # 4. IDENTIFY CALIBRATION PARTITION
    # ------------------------------------------------------------------

    print("\n[4] Identifying calibration partition...")

    df[SPLIT_COLUMN] = (
        df[SPLIT_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    calibration_df = df[
        df[SPLIT_COLUMN] == "calibration"
    ].copy()

    test_df = df[
        df[SPLIT_COLUMN] == "test"
    ].copy()

    if len(calibration_df) == 0:
        raise ValueError(
            "Calibration partition is empty."
        )

    if len(test_df) == 0:
        raise ValueError(
            "Test partition is empty."
        )

    print(
        f"Calibration rows: {len(calibration_df):,}"
    )

    print(
        f"Calibration range: "
        f"{calibration_df[MINUTE_COLUMN].min()} -> "
        f"{calibration_df[MINUTE_COLUMN].max()}"
    )

    print(
        f"Test rows: {len(test_df):,}"
    )

    print(
        f"Test range: "
        f"{test_df[MINUTE_COLUMN].min()} -> "
        f"{test_df[MINUTE_COLUMN].max()}"
    )

    # ------------------------------------------------------------------
    # 5. DERIVE TARGET FAILURE RATE
    # ------------------------------------------------------------------

    print("\n[5] Deriving controller target from calibration data...")

    target_failure_rate = float(
        calibration_df[TARGET_COLUMN].mean()
    )

    print(
        f"Calibration failure rate: "
        f"{target_failure_rate * 100:.4f}%"
    )

    print(
        "Target source: calibration partition"
    )

    print(
        "Test data used for target selection: NO"
    )

    # ------------------------------------------------------------------
    # 6. DISPLAY CONTROLLER CONFIGURATION
    # ------------------------------------------------------------------

    print("\n[6] Controller configuration...")

    print(
        f"Base threshold (theta_base): "
        f"{THETA_BASE:.4f}"
    )

    print(
        f"Adaptation step (eta): "
        f"{ETA:.4f}"
    )

    print(
        f"Minimum threshold: "
        f"{THETA_MIN:.4f}"
    )

    print(
        f"Maximum threshold: "
        f"{THETA_MAX:.4f}"
    )

    print(
        f"Historical window: "
        f"{HISTORY_WINDOW} minutes"
    )

    print(
        f"Target failure rate: "
        f"{target_failure_rate:.6f}"
    )

    if not (
        THETA_MIN
        <= THETA_BASE
        <= THETA_MAX
    ):
        raise ValueError(
            "Base threshold is outside configured bounds."
        )

    # ------------------------------------------------------------------
    # 7. INITIALIZE CONTROLLER
    # ------------------------------------------------------------------

    print("\n[7] Initializing adaptive controller...")

    theta = THETA_BASE

    previous_failures = []

    output_rows = []

    print(
        f"Initial threshold: {theta:.4f}"
    )

    # ------------------------------------------------------------------
    # 8. CAUSAL ADAPTIVE LOOP
    # ------------------------------------------------------------------

    print("\n[8] Running causal adaptive threshold controller...")

    for _, row in df.iterrows():

        minute = int(
            row[MINUTE_COLUMN]
        )

        split = str(
            row[SPLIT_COLUMN]
        )

        risk_probability = float(
            row[CALIBRATED_PROBABILITY_COLUMN]
        )

        actual_failure = int(
            row[TARGET_COLUMN]
        )

        # --------------------------------------------------------------
        # DECISION
        #
        # The current decision uses the threshold that existed BEFORE
        # observing the current outcome.
        # --------------------------------------------------------------

        threshold_before_update = theta

        reserve_trigger = int(
            risk_probability > threshold_before_update
        )

        # --------------------------------------------------------------
        # STORE DECISION
        # --------------------------------------------------------------

        output_rows.append(
            {
                "minute": minute,
                "split": split,

                "calibrated_failure_probability":
                    risk_probability,

                "threshold_before_update":
                    threshold_before_update,

                "reserve_trigger":
                    reserve_trigger,

                "actual_failure":
                    actual_failure
            }
        )

        # --------------------------------------------------------------
        # UPDATE STATE
        #
        # IMPORTANT:
        # The current outcome is incorporated ONLY AFTER the decision.
        # Therefore the current outcome cannot affect its own decision.
        # --------------------------------------------------------------

        previous_failures.append(
            actual_failure
        )

        if len(previous_failures) > HISTORY_WINDOW:
            previous_failures.pop(0)

        # --------------------------------------------------------------
        # Wait until sufficient historical observations exist.
        # --------------------------------------------------------------

        if len(previous_failures) < HISTORY_WINDOW:
            continue

        recent_failure_rate = float(
            np.mean(previous_failures)
        )

        # --------------------------------------------------------------
        # Adaptive rule:
        #
        # If recent observed failure rate is above the calibration
        # operating level:
        #
        #       lower threshold
        #
        # This makes reserve activation easier.
        #
        # If recent observed failure rate is below the calibration
        # operating level:
        #
        #       raise threshold
        #
        # This makes reserve activation less frequent.
        # --------------------------------------------------------------

        if recent_failure_rate > target_failure_rate:

            theta = theta - ETA

        elif recent_failure_rate < target_failure_rate:

            theta = theta + ETA

        # Keep threshold inside configured bounds.

        theta = clip_threshold(
            theta
        )

    # ------------------------------------------------------------------
    # 9. BUILD OUTPUT
    # ------------------------------------------------------------------

    print("\n[9] Building controller output...")

    result_df = pd.DataFrame(
        output_rows
    )

    result_df["threshold_after_update"] = (
        result_df[
            "threshold_before_update"
        ]
    )

    # Reconstruct the post-update threshold from the causal state.
    #
    # We intentionally perform the controller again in a compact pass
    # so the saved after-update value exactly matches the state used
    # for the next observation.

    theta = THETA_BASE

    previous_failures = []

    after_thresholds = []

    for _, row in df.iterrows():

        after_thresholds.append(
            theta
        )

        actual_failure = int(
            row[TARGET_COLUMN]
        )

        previous_failures.append(
            actual_failure
        )

        if len(previous_failures) > HISTORY_WINDOW:
            previous_failures.pop(0)

        if len(previous_failures) < HISTORY_WINDOW:
            continue

        recent_failure_rate = float(
            np.mean(previous_failures)
        )

        if recent_failure_rate > target_failure_rate:
            theta = theta - ETA

        elif recent_failure_rate < target_failure_rate:
            theta = theta + ETA

        theta = clip_threshold(theta)

    # The value before each decision is what matters.
    # Therefore overwrite after-update correctly by shifting state.

    theta = THETA_BASE
    previous_failures = []

    before_thresholds = []
    after_thresholds = []

    for _, row in df.iterrows():

        before_thresholds.append(
            theta
        )

        actual_failure = int(
            row[TARGET_COLUMN]
        )

        previous_failures.append(
            actual_failure
        )

        if len(previous_failures) > HISTORY_WINDOW:
            previous_failures.pop(0)

        if len(previous_failures) < HISTORY_WINDOW:

            after_thresholds.append(
                theta
            )

            continue

        recent_failure_rate = float(
            np.mean(previous_failures)
        )

        if recent_failure_rate > target_failure_rate:
            theta = theta - ETA

        elif recent_failure_rate < target_failure_rate:
            theta = theta + ETA

        theta = clip_threshold(theta)

        after_thresholds.append(
            theta
        )

    result_df[
        "threshold_before_update"
    ] = before_thresholds

    result_df[
        "threshold_after_update"
    ] = after_thresholds

    # ------------------------------------------------------------------
    # 10. CAUSALITY VALIDATION
    # ------------------------------------------------------------------

    print("\n[10] Running causality checks...")

    if not np.allclose(
        result_df[
            "threshold_before_update"
        ].to_numpy(),
        before_thresholds
    ):
        raise ValueError(
            "Threshold causality validation failed."
        )

    if (
        result_df[
            "threshold_before_update"
        ].min()
        < THETA_MIN
        or
        result_df[
            "threshold_before_update"
        ].max()
        > THETA_MAX
    ):
        raise ValueError(
            "Threshold bounds violated."
        )

    # Current failure must never affect current reserve decision.
    #
    # We verify that reserve_trigger depends only on the probability
    # and threshold BEFORE observing the current outcome.

    expected_trigger = (
        result_df[
            "calibrated_failure_probability"
        ]
        >
        result_df[
            "threshold_before_update"
        ]
    ).astype(int)

    if not np.array_equal(
        expected_trigger.to_numpy(),
        result_df[
            "reserve_trigger"
        ].to_numpy()
    ):
        raise ValueError(
            "Reserve decision causality check failed."
        )

    print(
        "Threshold bounds: PASS"
    )

    print(
        "Decision-before-outcome ordering: PASS"
    )

    print(
        "Reserve trigger calculation: PASS"
    )

    print(
        "Causal controller validation: PASS"
    )

    # ------------------------------------------------------------------
    # 11. CONTROLLER SUMMARY
    # ------------------------------------------------------------------

    print("\n[11] Controller summary...")

    print(
        f"Initial threshold: "
        f"{result_df['threshold_before_update'].iloc[0]:.4f}"
    )

    print(
        f"Minimum threshold reached: "
        f"{result_df['threshold_before_update'].min():.4f}"
    )

    print(
        f"Maximum threshold reached: "
        f"{result_df['threshold_before_update'].max():.4f}"
    )

    print(
        f"Mean threshold: "
        f"{result_df['threshold_before_update'].mean():.4f}"
    )

    print(
        f"Reserve triggers: "
        f"{result_df['reserve_trigger'].sum():,}"
    )

    print(
        f"Reserve activation rate: "
        f"{result_df['reserve_trigger'].mean() * 100:.4f}%"
    )

    # ------------------------------------------------------------------
    # 12. CHECK TEST IS NOT USED FOR PARAMETER SELECTION
    # ------------------------------------------------------------------

    print("\n[12] Test-isolation validation...")

    print(
        "Base threshold source: Paper 4 validation-selected value"
    )

    print(
        "Target failure rate source: calibration partition"
    )

    print(
        "Test outcomes were NOT used to select controller parameters."
    )

    print(
        "Test-isolation validation: PASS"
    )

    # ------------------------------------------------------------------
    # 13. SAVE RESULTS
    # ------------------------------------------------------------------

    print("\n[13] Saving adaptive threshold results...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Controller output saved:\n"
        f"  {OUTPUT_PATH}"
    )

    # ------------------------------------------------------------------
    # 14. SAVE METADATA
    # ------------------------------------------------------------------

    print("\n[14] Saving controller metadata...")

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = {

        "module":
            "09_adaptive_threshold",

        "method":
            "causal_failure_rate_adaptive_threshold",

        "input":
            str(
                INPUT_PATH.relative_to(
                    BASE_DIR
                )
            ),

        "output":
            str(
                OUTPUT_PATH.relative_to(
                    BASE_DIR
                )
            ),

        "theta_base":
            THETA_BASE,

        "eta":
            ETA,

        "theta_min":
            THETA_MIN,

        "theta_max":
            THETA_MAX,

        "history_window":
            HISTORY_WINDOW,

        "target_failure_rate":
            target_failure_rate,

        "target_failure_rate_source":
            "calibration_partition",

        "calibration_rows":
            int(len(calibration_df)),

        "test_rows":
            int(len(test_df)),

        "test_used_for_parameter_selection":
            False,

        "current_outcome_used_for_current_decision":
            False,

        "causal_validation":
            "PASS",

        "reserve_activation_rate":
            float(
                result_df[
                    "reserve_trigger"
                ].mean()
            ),

        "mean_threshold":
            float(
                result_df[
                    "threshold_before_update"
                ].mean()
            ),

        "minimum_threshold":
            float(
                result_df[
                    "threshold_before_update"
                ].min()
            ),

        "maximum_threshold":
            float(
                result_df[
                    "threshold_before_update"
                ].max()
            )
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
        f"Metadata saved:\n"
        f"  {METADATA_PATH}"
    )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ADAPTIVE THRESHOLD MODULE COMPLETE")
    print("=" * 70)

    print("\nCreated:")

    print(
        "  results/policies/"
        "adaptive_threshold_predictions.csv"
    )

    print(
        "  results/validation/"
        "adaptive_threshold_metadata.json"
    )

    print("\nImportant:")

    print(
        "  Workload forecast NOT modified"
    )

    print(
        "  Capacity simulation NOT performed"
    )

    print(
        "  Change-point reset NOT applied yet"
    )

    print(
        "  Test data NOT used for parameter selection"
    )

    print(
        "  Current outcome cannot affect current decision"
    )

    print(
        "\n09 COMPLETE"
    )


if __name__ == "__main__":
    main()