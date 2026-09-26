from pathlib import Path
import itertools
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "burstgpt_5min.csv"

OUTPUT_DIR = ROOT / "results" / "sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = (
    OUTPUT_DIR / "change_point_sensitivity.csv"
)

OUTPUT_ALARMS = (
    OUTPUT_DIR / "change_point_sensitivity_alarms.csv"
)

OUTPUT_META = (
    ROOT
    / "results"
    / "validation"
    / "change_point_sensitivity_metadata.json"
)


# ============================================================
# VALIDATED MODULE 03 BASELINE
# ============================================================

BASELINE = {
    "short_window": 12,
    "long_window": 276,
    "absolute_log_shift": 0.10,
    "min_alarm_gap": 288,
    "threshold_quantile": 0.95,
}


# ============================================================
# SENSITIVITY GRID
# ============================================================

SHORT_WINDOWS = [6, 12, 24]
LONG_WINDOWS = [144, 276, 432]
ABSOLUTE_LOG_SHIFTS = [0.05, 0.10, 0.20]
MIN_ALARM_GAPS = [144, 288, 576]

TRAINING_SIZE = 12384


# ============================================================
# EXACT MODULE 03 SCORE CALCULATION
# ============================================================

def calculate_change_scores(
    workload,
    short_window,
    long_window,
):
    """
    Reproduce the exact window indexing used by
    validated Module 03.

    IMPORTANT:
    The current observation t IS included in the
    short window, exactly as Module 03 does.

    Short:
        [t-short_window+1 ... t]

    Long:
        [t-short_window-long_window+1
         ...
         t-short_window]

    This intentionally preserves Module 03
    implementation for reproducibility.
    """

    workload = np.asarray(
        workload,
        dtype=float,
    )

    z = np.log1p(workload)

    n = len(z)

    change_score = np.full(
        n,
        np.nan,
    )

    short_level = np.full(
        n,
        np.nan,
    )

    long_level = np.full(
        n,
        np.nan,
    )

    robust_scale = np.full(
        n,
        np.nan,
    )

    level_difference = np.full(
        n,
        np.nan,
    )

    short_long_ratio = np.full(
        n,
        np.nan,
    )

    first_valid_index = (
        long_window
        + short_window
        - 1
    )

    for t in range(
        first_valid_index,
        n,
    ):

        # ----------------------------------------------------
        # EXACT MODULE 03 SHORT WINDOW
        # ----------------------------------------------------

        short_start = (
            t
            - short_window
            + 1
        )

        short_end = t + 1

        short_values = z[
            short_start:short_end
        ]

        # ----------------------------------------------------
        # EXACT MODULE 03 LONG WINDOW
        # ----------------------------------------------------

        long_start = (
            t
            - short_window
            - long_window
            + 1
        )

        long_end = (
            t
            - short_window
            + 1
        )

        long_values = z[
            long_start:long_end
        ]

        # ----------------------------------------------------
        # STATISTICS
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
                25,
            )
        )

        q75 = float(
            np.percentile(
                long_values,
                75,
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
        # EXACT MODULE 03 ZERO-SCALE HANDLING
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # EXACT MODULE 03 RATIO
        # ----------------------------------------------------

        ratio = np.exp(
            short_mean
            - long_median
        )

        change_score[t] = score

        short_level[t] = short_mean

        long_level[t] = long_median

        robust_scale[t] = scale

        level_difference[t] = difference

        short_long_ratio[t] = ratio

    return {
        "change_score": change_score,
        "short_level": short_level,
        "long_level": long_level,
        "robust_scale": robust_scale,
        "absolute_log_shift": level_difference,
        "short_long_ratio": short_long_ratio,
    }


# ============================================================
# TRAINING THRESHOLD
# ============================================================

def derive_training_threshold(
    workload,
    short_window,
    long_window,
    quantile,
):
    """
    Exact Module 03 threshold procedure.

    Threshold is calculated from training data only.
    """

    training_workload = np.asarray(
        workload[:TRAINING_SIZE],
        dtype=float,
    )

    statistics = calculate_change_scores(
        training_workload,
        short_window,
        long_window,
    )

    scores = statistics[
        "change_score"
    ]

    scores = scores[
        np.isfinite(scores)
    ]

    if len(scores) == 0:
        raise ValueError(
            "No valid training scores."
        )

    return float(
        np.quantile(
            scores,
            quantile,
        )
    )


# ============================================================
# ALARM DETECTION
# ============================================================

def detect_alarms(
    statistics,
    threshold,
    absolute_log_shift,
    min_alarm_gap,
):
    """
    Exact Module 03 alarm logic.
    """

    scores = statistics[
        "change_score"
    ]

    differences = statistics[
        "absolute_log_shift"
    ]

    n = len(scores)

    alarms = np.zeros(
        n,
        dtype=np.int8,
    )

    last_alarm = -10**9

    for t in range(n):

        if not np.isfinite(
            scores[t]
        ):
            continue

        score_condition = (
            scores[t] >= threshold
        )

        level_condition = (
            differences[t]
            >= absolute_log_shift
        )

        spacing_condition = (
            (t - last_alarm)
            >= min_alarm_gap
        )

        alarm = (
            score_condition
            and level_condition
            and spacing_condition
        )

        if alarm:

            alarms[t] = 1

            last_alarm = t

    return alarms


# ============================================================
# ALARM TIMING COMPARISON
# ============================================================

def compare_alarm_timing(
    baseline_indices,
    candidate_indices,
    tolerance=12,
):
    """
    Descriptive comparison only.

    tolerance=12 five-minute bins = 1 hour.
    """

    baseline_indices = np.asarray(
        baseline_indices,
        dtype=int,
    )

    candidate_indices = np.asarray(
        candidate_indices,
        dtype=int,
    )

    if len(baseline_indices) == 0:

        return {
            "baseline_match_rate": np.nan,
            "candidate_match_rate": (
                1.0
                if len(candidate_indices) == 0
                else 0.0
            ),
            "matched_alarm_count": 0,
        }

    if len(candidate_indices) == 0:

        return {
            "baseline_match_rate": 0.0,
            "candidate_match_rate": np.nan,
            "matched_alarm_count": 0,
        }

    baseline_matches = sum(
        np.any(
            np.abs(
                candidate_indices - b
            )
            <= tolerance
        )
        for b in baseline_indices
    )

    candidate_matches = sum(
        np.any(
            np.abs(
                baseline_indices - c
            )
            <= tolerance
        )
        for c in candidate_indices
    )

    return {
        "baseline_match_rate": (
            baseline_matches
            / len(baseline_indices)
        ),
        "candidate_match_rate": (
            candidate_matches
            / len(candidate_indices)
        ),
        "matched_alarm_count": int(
            min(
                baseline_matches,
                candidate_matches,
            )
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "MODULE 17 — CHANGE-POINT SENSITIVITY ANALYSIS"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    required = {
        "interval",
        "total_tokens",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    df = (
        df
        .sort_values("interval")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # INTERVAL VALIDATION
    # --------------------------------------------------------

    expected = np.arange(
        len(df)
    )

    if not np.array_equal(
        df["interval"].to_numpy(),
        expected,
    ):
        raise ValueError(
            "Interval sequence is not "
            "continuous from 0."
        )

    print(
        "Interval ordering: PASS"
    )

    workload = (
        df["total_tokens"]
        .to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    print(
        "\nRunning exact validated "
        "Module 03 baseline..."
    )

    baseline_statistics = (
        calculate_change_scores(
            workload,
            BASELINE[
                "short_window"
            ],
            BASELINE[
                "long_window"
            ],
        )
    )

    baseline_threshold = (
        derive_training_threshold(
            workload,
            BASELINE[
                "short_window"
            ],
            BASELINE[
                "long_window"
            ],
            BASELINE[
                "threshold_quantile"
            ],
        )
    )

    baseline_alarms = detect_alarms(
        baseline_statistics,
        baseline_threshold,
        BASELINE[
            "absolute_log_shift"
        ],
        BASELINE[
            "min_alarm_gap"
        ],
    )

    baseline_indices = np.where(
        baseline_alarms == 1
    )[0]

    print(
        f"Baseline threshold: "
        f"{baseline_threshold:.6f}"
    )

    print(
        f"Baseline alarms: "
        f"{len(baseline_indices)}"
    )

    # --------------------------------------------------------
    # HARD REPRODUCTION CHECK
    # --------------------------------------------------------

    EXPECTED_THRESHOLD = 5.088177
    EXPECTED_ALARMS = 27

    threshold_difference = abs(
        baseline_threshold
        - EXPECTED_THRESHOLD
    )

    if (
        threshold_difference > 0.001
        or
        len(baseline_indices)
        != EXPECTED_ALARMS
    ):

        raise RuntimeError(
            "\n"
            "BASELINE REPRODUCTION FAILED.\n"
            f"Expected threshold ≈ "
            f"{EXPECTED_THRESHOLD:.6f}; "
            f"got {baseline_threshold:.6f}.\n"
            f"Expected alarms = "
            f"{EXPECTED_ALARMS}; "
            f"got {len(baseline_indices)}.\n"
            "\n"
            "Module 17 is NOT accepted."
        )

    # --------------------------------------------------------
    # BASELINE SPLIT
    # --------------------------------------------------------

    baseline_train = int(
        np.sum(
            baseline_indices
            < 12384
        )
    )

    baseline_calibration = int(
        np.sum(
            (
                baseline_indices
                >= 12384
            )
            &
            (
                baseline_indices
                < 14976
            )
        )
    )

    baseline_test = int(
        np.sum(
            baseline_indices
            >= 14976
        )
    )

    print(
        "Baseline split:"
    )

    print(
        f"  Train: {baseline_train}"
    )

    print(
        f"  Calibration: "
        f"{baseline_calibration}"
    )

    print(
        f"  Test: {baseline_test}"
    )

    if (
        baseline_train != 22
        or baseline_calibration != 3
        or baseline_test != 2
    ):

        raise RuntimeError(
            "Baseline alarm split does not "
            "match validated Module 03."
        )

    print(
        "Baseline reproduction: PASS"
    )

    # --------------------------------------------------------
    # CONFIGURATION GRID
    # --------------------------------------------------------

    configurations = []

    for (
        short_window,
        long_window,
        absolute_log_shift,
        min_alarm_gap,
    ) in itertools.product(
        SHORT_WINDOWS,
        LONG_WINDOWS,
        ABSOLUTE_LOG_SHIFTS,
        MIN_ALARM_GAPS,
    ):

        if long_window <= short_window:
            continue

        configurations.append({
            "short_window": short_window,
            "long_window": long_window,
            "absolute_log_shift": (
                absolute_log_shift
            ),
            "min_alarm_gap": (
                min_alarm_gap
            ),
            "threshold_quantile": 0.95,
        })

    print(
        f"\nSensitivity configurations: "
        f"{len(configurations)}"
    )

    # --------------------------------------------------------
    # RUN SENSITIVITY
    # --------------------------------------------------------

    summary_rows = []
    alarm_rows = []

    for config_id, config in enumerate(
        configurations,
        start=1,
    ):

        statistics = (
            calculate_change_scores(
                workload,
                config[
                    "short_window"
                ],
                config[
                    "long_window"
                ],
            )
        )

        threshold = (
            derive_training_threshold(
                workload,
                config[
                    "short_window"
                ],
                config[
                    "long_window"
                ],
                config[
                    "threshold_quantile"
                ],
            )
        )

        alarms = detect_alarms(
            statistics,
            threshold,
            config[
                "absolute_log_shift"
            ],
            config[
                "min_alarm_gap"
            ],
        )

        alarm_indices = np.where(
            alarms == 1
        )[0]

        # ----------------------------------------------------
        # ALARM GAPS
        # ----------------------------------------------------

        if len(alarm_indices) >= 2:

            minimum_gap = int(
                np.min(
                    np.diff(
                        alarm_indices
                    )
                )
            )

        else:

            minimum_gap = np.nan

        # ----------------------------------------------------
        # SPLIT COUNTS
        # ----------------------------------------------------

        train_count = int(
            np.sum(
                alarm_indices
                < 12384
            )
        )

        calibration_count = int(
            np.sum(
                (
                    alarm_indices
                    >= 12384
                )
                &
                (
                    alarm_indices
                    < 14976
                )
            )
        )

        test_count = int(
            np.sum(
                alarm_indices
                >= 14976
            )
        )

        # ----------------------------------------------------
        # BASELINE COMPARISON
        # ----------------------------------------------------

        overlap = (
            compare_alarm_timing(
                baseline_indices,
                alarm_indices,
                tolerance=12,
            )
        )

        is_baseline = (
            config[
                "short_window"
            ]
            == BASELINE[
                "short_window"
            ]
            and
            config[
                "long_window"
            ]
            == BASELINE[
                "long_window"
            ]
            and
            config[
                "absolute_log_shift"
            ]
            == BASELINE[
                "absolute_log_shift"
            ]
            and
            config[
                "min_alarm_gap"
            ]
            == BASELINE[
                "min_alarm_gap"
            ]
        )

        summary_rows.append({
            "config_id": config_id,
            "is_baseline": is_baseline,
            **config,
            "derived_threshold": (
                threshold
            ),
            "total_alarms": (
                len(alarm_indices)
            ),
            "training_alarms": (
                train_count
            ),
            "calibration_alarms": (
                calibration_count
            ),
            "test_alarms": (
                test_count
            ),
            "minimum_observed_alarm_gap": (
                minimum_gap
            ),
            **overlap,
        })

        # ----------------------------------------------------
        # ALARM DETAILS
        # ----------------------------------------------------

        for alarm_index in alarm_indices:

            if alarm_index < 12384:
                split = "train"

            elif alarm_index < 14976:
                split = "calibration"

            else:
                split = "test"

            alarm_rows.append({
                "config_id": config_id,
                "interval": int(
                    alarm_index
                ),
                "split": split,
                "change_score": float(
                    statistics[
                        "change_score"
                    ][alarm_index]
                ),
                "derived_threshold": float(
                    threshold
                ),
                "short_level_log": float(
                    statistics[
                        "short_level"
                    ][alarm_index]
                ),
                "long_level_log": float(
                    statistics[
                        "long_level"
                    ][alarm_index]
                ),
                "robust_scale": float(
                    statistics[
                        "robust_scale"
                    ][alarm_index]
                ),
                "absolute_log_shift": float(
                    statistics[
                        "absolute_log_shift"
                    ][alarm_index]
                ),
                "short_long_ratio": float(
                    statistics[
                        "short_long_ratio"
                    ][alarm_index]
                ),
            })

        print(
            f"[{config_id:02d}/"
            f"{len(configurations)}] "
            f"S={config['short_window']} "
            f"L={config['long_window']} "
            f"Shift="
            f"{config['absolute_log_shift']:.2f} "
            f"Gap={config['min_alarm_gap']} "
            f"-> threshold="
            f"{threshold:.4f}, "
            f"alarms="
            f"{len(alarm_indices)}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summary_rows
    )

    alarm_df = pd.DataFrame(
        alarm_rows
    )

    summary_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    alarm_df.to_csv(
        OUTPUT_ALARMS,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    baseline_rows = summary_df[
        summary_df[
            "is_baseline"
        ]
    ]

    if len(baseline_rows) != 1:
        raise RuntimeError(
            "Baseline configuration was "
            "not uniquely identified."
        )

    row = baseline_rows.iloc[0]

    if int(
        row["total_alarms"]
    ) != 27:

        raise RuntimeError(
            "Baseline alarm count mismatch."
        )

    if (
        int(row["training_alarms"])
        != 22
        or
        int(row["calibration_alarms"])
        != 3
        or
        int(row["test_alarms"])
        != 2
    ):

        raise RuntimeError(
            "Baseline alarm split mismatch."
        )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = {
        "module": (
            "17_change_point_sensitivity"
        ),
        "source_implementation": (
            "validated Module 03"
        ),
        "implementation_note": (
            "Sensitivity analysis preserves "
            "the exact Module 03 window indexing, "
            "including the current observation "
            "in the short window."
        ),
        "rows": int(
            len(df)
        ),
        "training_size": (
            TRAINING_SIZE
        ),
        "baseline": BASELINE,
        "baseline_threshold": (
            baseline_threshold
        ),
        "baseline_alarm_count": (
            len(baseline_indices)
        ),
        "baseline_split": {
            "train": baseline_train,
            "calibration": (
                baseline_calibration
            ),
            "test": baseline_test,
        },
        "test_used_for_parameter_selection": (
            False
        ),
        "configurations_evaluated": (
            len(configurations)
        ),
        "validation": "PASS",
    }

    OUTPUT_META.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_META,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CHANGE-POINT SENSITIVITY ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Configurations evaluated: "
        f"{len(configurations)}"
    )

    print(
        f"Baseline threshold: "
        f"{baseline_threshold:.6f}"
    )

    print(
        f"Baseline alarms: "
        f"{len(baseline_indices)}"
    )

    print(
        "Baseline split: "
        f"{baseline_train} train / "
        f"{baseline_calibration} calibration / "
        f"{baseline_test} test"
    )

    print(
        "\nBaseline reproduction: PASS"
    )

    print("\nOutputs:")
    print(
        f"  {OUTPUT_CSV}"
    )
    print(
        f"  {OUTPUT_ALARMS}"
    )
    print(
        f"  {OUTPUT_META}"
    )

    print(
        "\nValidation: PASS"
    )


if __name__ == "__main__":
    main()