"""
03_change_point_detector.py

Paper 3 change-point detector.

Input:
    data/burstgpt_5min.csv

Output:
    results/alarms/change_point_5min.csv
    results/alarms/change_point_metadata.json

Method:
    z_t = log(1 + y_t)

    Short window:
        latest 12 five-minute observations

    Long window:
        preceding 276 five-minute observations

    Robust scale:
        IQR / 1.349

    Change score:
        |short_mean - long_median| / robust_scale

    Alarm:
        score >= training 95th percentile
        AND absolute log-level shift >= 0.10
        AND at least 288 bins since previous alarm

Important:
    The threshold is DERIVED from the training data.
    We do not simply hard-code the value reported by Paper 3.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/burstgpt_5min.csv"
)

OUTPUT_DIR = Path(
    "results/alarms"
)

OUTPUT_FILE = OUTPUT_DIR / (
    "change_point_5min.csv"
)

METADATA_FILE = OUTPUT_DIR / (
    "change_point_metadata.json"
)


# Paper 3 parameters
SHORT_WINDOW = 12
LONG_WINDOW = 276

ABSOLUTE_LOG_SHIFT = 0.10

MIN_ALARM_GAP = 288

TRAINING_SIZE = 12384

THRESHOLD_QUANTILE = 0.95

PAPER3_REPORTED_THRESHOLD = 5.067


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n[1] Loading five-minute workload...")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nInput file not found:\n"
            f"{INPUT_FILE.resolve()}\n\n"
            f"Run 02_build_time_views.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "interval",
        "total_tokens",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(
        f"Rows loaded: {len(df):,}"
    )

    return df


# ============================================================
# BUILD CHANGE-POINT SCORES
# ============================================================

def calculate_change_point_statistics(
    workload,
    threshold
):

    """
    Calculate the Paper 3 causal change-point statistics.

    At time t:

        short window =
            observations [t-11 ... t]

        long window =
            observations [t-287 ... t-12]

    Therefore the long window does not overlap
    with the short window.
    """

    workload = np.asarray(
        workload,
        dtype=float
    )

    z = np.log1p(workload)

    n = len(z)

    change_score = np.full(
        n,
        np.nan
    )

    change_alarm = np.zeros(
        n,
        dtype=np.int8
    )

    short_level = np.full(
        n,
        np.nan
    )

    long_level = np.full(
        n,
        np.nan
    )

    robust_scale = np.full(
        n,
        np.nan
    )

    level_difference = np.full(
        n,
        np.nan
    )

    short_long_ratio = np.full(
        n,
        np.nan
    )

    elapsed_since_alarm = np.full(
        n,
        np.nan
    )

    last_alarm = -10**9

    first_valid_index = (
        LONG_WINDOW
        + SHORT_WINDOW
        - 1
    )

    for t in range(
        first_valid_index,
        n
    ):

        # ----------------------------------------------------
        # Short window
        # ----------------------------------------------------

        short_start = (
            t
            - SHORT_WINDOW
            + 1
        )

        short_end = t + 1

        short_values = z[
            short_start:short_end
        ]

        # ----------------------------------------------------
        # Long window
        # ----------------------------------------------------

        long_start = (
            t
            - SHORT_WINDOW
            - LONG_WINDOW
            + 1
        )

        long_end = (
            t
            - SHORT_WINDOW
            + 1
        )

        long_values = z[
            long_start:long_end
        ]

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        short_mean = float(
            np.mean(short_values)
        )

        long_median = float(
            np.median(long_values)
        )

        q25 = float(
            np.percentile(
                long_values,
                25
            )
        )

        q75 = float(
            np.percentile(
                long_values,
                75
            )
        )

        scale = (
            q75 - q25
        ) / 1.349

        difference = abs(
            short_mean
            - long_median
        )

        # ----------------------------------------------------
        # Robust change score
        # ----------------------------------------------------

        if scale > 0:

            score = (
                difference
                / scale
            )

        else:

            # If historical variation is exactly zero:
            #
            # no level change -> score 0
            # level change    -> infinite score

            if difference > 0:
                score = np.inf
            else:
                score = 0.0

        # ----------------------------------------------------
        # Ratio
        # ----------------------------------------------------

        ratio = np.exp(
            short_mean
            - long_median
        )

        # ----------------------------------------------------
        # Candidate alarm
        # ----------------------------------------------------

        score_condition = (
            score >= threshold
        )

        level_condition = (
            difference
            >= ABSOLUTE_LOG_SHIFT
        )

        spacing_condition = (
            (t - last_alarm)
            >= MIN_ALARM_GAP
        )

        alarm = (
            score_condition
            and level_condition
            and spacing_condition
        )

        # ----------------------------------------------------
        # Store statistics
        # ----------------------------------------------------

        change_score[t] = score

        short_level[t] = (
            short_mean
        )

        long_level[t] = (
            long_median
        )

        robust_scale[t] = (
            scale
        )

        level_difference[t] = (
            difference
        )

        short_long_ratio[t] = (
            ratio
        )

        elapsed_since_alarm[t] = (
            t - last_alarm
        )

        # ----------------------------------------------------
        # Register alarm
        # ----------------------------------------------------

        if alarm:

            change_alarm[t] = 1

            last_alarm = t

    return pd.DataFrame(
        {
            "change_score":
                change_score,

            "change_alarm":
                change_alarm,

            "short_level_log":
                short_level,

            "long_level_log":
                long_level,

            "robust_scale":
                robust_scale,

            "absolute_log_shift":
                level_difference,

            "short_long_ratio":
                short_long_ratio,

            "elapsed_bins_since_alarm":
                elapsed_since_alarm,
        }
    )


# ============================================================
# DERIVE TRAINING THRESHOLD
# ============================================================

def derive_training_threshold(
    workload
):

    """
    Derive the 95th percentile threshold using
    only the Paper 3 training block.

    This prevents test-set information from
    influencing the detector threshold.
    """

    print(
        "\n[2] Deriving change-point threshold..."
    )

    if len(workload) < TRAINING_SIZE:

        raise ValueError(
            "Dataset is shorter than the "
            "Paper 3 training block."
        )

    training_workload = np.asarray(
        workload[
            :TRAINING_SIZE
        ],
        dtype=float
    )

    z = np.log1p(
        training_workload
    )

    scores = []

    first_valid_index = (
        LONG_WINDOW
        + SHORT_WINDOW
        - 1
    )

    for t in range(
        first_valid_index,
        TRAINING_SIZE
    ):

        short_values = z[
            t - SHORT_WINDOW + 1:
            t + 1
        ]

        long_values = z[
            t - SHORT_WINDOW - LONG_WINDOW + 1:
            t - SHORT_WINDOW + 1
        ]

        short_mean = float(
            np.mean(short_values)
        )

        long_median = float(
            np.median(long_values)
        )

        q25 = float(
            np.percentile(
                long_values,
                25
            )
        )

        q75 = float(
            np.percentile(
                long_values,
                75
            )
        )

        scale = (
            q75 - q25
        ) / 1.349

        difference = abs(
            short_mean
            - long_median
        )

        if scale > 0:

            score = (
                difference
                / scale
            )

        else:

            if difference > 0:
                score = np.inf
            else:
                score = 0.0

        if np.isfinite(score):

            scores.append(score)

    if not scores:

        raise ValueError(
            "Could not calculate any "
            "training change scores."
        )

    threshold = float(
        np.quantile(
            scores,
            THRESHOLD_QUANTILE
        )
    )

    print(
        f"Training observations: "
        f"{TRAINING_SIZE:,}"
    )

    print(
        f"Valid training scores: "
        f"{len(scores):,}"
    )

    print(
        f"Derived 95th percentile: "
        f"{threshold:.6f}"
    )

    print(
        f"Paper 3 reported threshold: "
        f"{PAPER3_REPORTED_THRESHOLD:.3f}"
    )

    print(
        f"Difference from reported value: "
        f"{threshold - PAPER3_REPORTED_THRESHOLD:.6f}"
    )

    return threshold


# ============================================================
# VALIDATE DETECTOR
# ============================================================

def validate_detector(
    result
):

    print(
        "\n[4] Validating detector output..."
    )

    # --------------------------------------------------------
    # Alarm values
    # --------------------------------------------------------

    unique_alarm_values = sorted(
        result["change_alarm"]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        "Alarm values:",
        unique_alarm_values
    )

    if not set(
        unique_alarm_values
    ).issubset({0, 1}):

        raise ValueError(
            "change_alarm contains "
            "values other than 0/1."
        )

    # --------------------------------------------------------
    # Alarm positions
    # --------------------------------------------------------

    alarm_positions = (
        result.index[
            result["change_alarm"] == 1
        ]
        .to_numpy()
    )

    print(
        f"Total alarms: "
        f"{len(alarm_positions):,}"
    )

    # --------------------------------------------------------
    # Minimum gap
    # --------------------------------------------------------

    if len(alarm_positions) > 1:

        gaps = np.diff(
            alarm_positions
        )

        minimum_gap = int(
            gaps.min()
        )

        print(
            f"Minimum alarm gap: "
            f"{minimum_gap} bins"
        )

        if minimum_gap < MIN_ALARM_GAP:

            raise ValueError(
                "Alarm spacing constraint "
                "was violated."
            )

    else:

        print(
            "Minimum alarm gap: "
            "not applicable"
        )

    # --------------------------------------------------------
    # Training threshold leakage check
    # --------------------------------------------------------

    print(
        f"Training boundary: "
        f"index {TRAINING_SIZE - 1}"
    )

    print(
        "Threshold source: "
        "training block only"
    )

    print(
        "Detector validation: PASS"
    )


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_outputs(
    original_df,
    detector_df,
    threshold
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Combined output
    # --------------------------------------------------------

    result = pd.concat(
        [
            original_df.reset_index(
                drop=True
            ),
            detector_df.reset_index(
                drop=True
            ),
        ],
        axis=1
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    alarm_positions = (
        result.index[
            result["change_alarm"] == 1
        ]
        .tolist()
    )

    metadata = {

        "input_file":
            str(INPUT_FILE),

        "output_file":
            str(OUTPUT_FILE),

        "training_size":
            TRAINING_SIZE,

        "short_window_bins":
            SHORT_WINDOW,

        "long_window_bins":
            LONG_WINDOW,

        "absolute_log_shift":
            ABSOLUTE_LOG_SHIFT,

        "minimum_alarm_gap_bins":
            MIN_ALARM_GAP,

        "threshold_quantile":
            THRESHOLD_QUANTILE,

        "derived_threshold":
            threshold,

        "paper3_reported_threshold":
            PAPER3_REPORTED_THRESHOLD,

        "alarm_count":
            len(alarm_positions),

        "alarm_positions":
            alarm_positions,
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "PAPER 3 CHANGE-POINT DETECTOR"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    workload = (
        df["total_tokens"]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Derive threshold
    # --------------------------------------------------------

    threshold = (
        derive_training_threshold(
            workload
        )
    )

    # --------------------------------------------------------
    # Calculate detector
    # --------------------------------------------------------

    print(
        "\n[3] Running change-point detector..."
    )

    detector_df = (
        calculate_change_point_statistics(
            workload,
            threshold
        )
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    result = pd.concat(
        [
            df.reset_index(drop=True),
            detector_df.reset_index(drop=True),
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_detector(
        result
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result = save_outputs(
        df,
        detector_df,
        threshold
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    alarm_count = int(
        result["change_alarm"].sum()
    )

    valid_scores = int(
        result["change_score"]
        .notna()
        .sum()
    )

    print(
        "\n" + "-" * 70
    )

    print(
        "CHANGE-POINT SUMMARY"
    )

    print(
        "-" * 70
    )

    print(
        f"Five-minute observations : "
        f"{len(result):,}"
    )

    print(
        f"Valid detector positions : "
        f"{valid_scores:,}"
    )

    print(
        f"Derived threshold        : "
        f"{threshold:.6f}"
    )

    print(
        f"Total alarms             : "
        f"{alarm_count:,}"
    )

    print(
        "\nOutput files:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        f"  {METADATA_FILE}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CHANGE-POINT DETECTOR COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()