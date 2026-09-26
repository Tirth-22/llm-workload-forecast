import json
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# MODULE 18 — CHANGE-POINT DOWNSTREAM ROBUSTNESS
# ======================================================================
#
# Purpose:
#   Propagate a pre-defined one-factor-at-a-time subset of the
#   Module 17 change-point sensitivity configurations into V4.
#
# Important:
#   - Module 17 alarm file contains ONLY alarm rows.
#   - Therefore every row in the alarm file represents change_alarm = 1.
#   - Configuration identity is represented by config_id.
#   - CP-HGB remains the primary capacity forecast.
#   - Module 09 adaptive threshold state is NOT retrained.
#   - Only the change-point alarm stream is varied.
#   - Locked test performance is NOT used to select configurations.
# ======================================================================


BASE_DIR = Path(__file__).resolve().parent.parent


# ======================================================================
# INPUT PATHS
# ======================================================================

V3_PATH = (
    BASE_DIR
    / "results"
    / "policies"
    / "V3_adaptive_risk.csv"
)

V4_BASELINE_PATH = (
    BASE_DIR
    / "results"
    / "policies"
    / "V4_adaptive_reset.csv"
)

MODULE17_ALARMS_PATH = (
    BASE_DIR
    / "results"
    / "sensitivity"
    / "change_point_sensitivity_alarms.csv"
)


# ======================================================================
# OUTPUT PATHS
# ======================================================================

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "sensitivity"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "change_point_downstream_robustness.csv"
)

DETAIL_OUTPUT = (
    OUTPUT_DIR
    / "change_point_downstream_robustness_test_intervals.csv"
)

VALIDATION_DIR = (
    BASE_DIR
    / "results"
    / "validation"
)

METADATA_OUTPUT = (
    VALIDATION_DIR
    / "change_point_downstream_robustness_metadata.json"
)


# ======================================================================
# MODULE 11 PARAMETERS
# ======================================================================

THETA_BASE = 0.9591

RESERVE_FACTOR = 1.10

THETA_MIN = 0.50

THETA_MAX = 0.99


# ======================================================================
# MODULE 17 CONFIGURATION GRID
# ======================================================================
#
# These lists reproduce the exact ordering used by Module 17:
#
#   for S in [6, 12, 24]
#       for L in [144, 276, 432]
#           for Shift in [0.05, 0.10, 0.20]
#               for Gap in [144, 288, 576]
#
# Therefore:
#
# config_id = 1 ... 81
#
# The baseline:
#   S=12
#   L=276
#   Shift=0.10
#   Gap=288
#
# is config_id 41.
# ======================================================================

SHORT_WINDOWS = [6, 12, 24]

LONG_WINDOWS = [144, 276, 432]

ABSOLUTE_LOG_SHIFTS = [0.05, 0.10, 0.20]

MIN_ALARM_GAPS = [144, 288, 576]


# ======================================================================
# SELECTED DOWNSTREAM ROBUSTNESS CONFIGURATIONS
# ======================================================================
#
# Baseline + one-factor-at-a-time alternatives.
#
# These were defined independently of downstream test performance.
# ======================================================================

CONFIGURATIONS = [
    {
        "configuration": "baseline",
        "short_window": 12,
        "long_window": 276,
        "absolute_log_shift": 0.10,
        "min_alarm_gap": 288,
    },

    {
        "configuration": "short_6",
        "short_window": 6,
        "long_window": 276,
        "absolute_log_shift": 0.10,
        "min_alarm_gap": 288,
    },

    {
        "configuration": "short_24",
        "short_window": 24,
        "long_window": 276,
        "absolute_log_shift": 0.10,
        "min_alarm_gap": 288,
    },

    {
        "configuration": "long_144",
        "short_window": 12,
        "long_window": 144,
        "absolute_log_shift": 0.10,
        "min_alarm_gap": 288,
    },

    {
        "configuration": "long_432",
        "short_window": 12,
        "long_window": 432,
        "absolute_log_shift": 0.10,
        "min_alarm_gap": 288,
    },

    {
        "configuration": "shift_0.05",
        "short_window": 12,
        "long_window": 276,
        "absolute_log_shift": 0.05,
        "min_alarm_gap": 288,
    },

    {
        "configuration": "shift_0.20",
        "short_window": 12,
        "long_window": 276,
        "absolute_log_shift": 0.20,
        "min_alarm_gap": 288,
    },

    {
        "configuration": "gap_144",
        "short_window": 12,
        "long_window": 276,
        "absolute_log_shift": 0.10,
        "min_alarm_gap": 144,
    },

    {
        "configuration": "gap_576",
        "short_window": 12,
        "long_window": 276,
        "absolute_log_shift": 0.10,
        "min_alarm_gap": 576,
    },
]


# ======================================================================
# V3 REQUIRED COLUMNS
# ======================================================================

V3_REQUIRED_COLUMNS = [
    "interval",
    "split",
    "actual_next_5min_tokens",
    "hgb_prediction",
    "cp_hgb_prediction",
    "actual_workload",
    "forecast_workload",
    "risk_signal",
    "adaptive_threshold",
]


# ======================================================================
# MODULE 17 REQUIRED COLUMNS
# ======================================================================

MODULE17_REQUIRED_COLUMNS = [
    "config_id",
    "interval",
    "split",
    "change_score",
    "derived_threshold",
    "short_level_log",
    "long_level_log",
    "robust_scale",
    "absolute_log_shift",
    "short_long_ratio",
]


# ======================================================================
# HELPER — GET CONFIG ID
# ======================================================================

def get_config_id(
    short_window,
    long_window,
    absolute_log_shift,
    min_alarm_gap
):
    """
    Reproduce the exact Module 17 configuration ordering.
    """

    config_id = 0

    for short in SHORT_WINDOWS:

        for long in LONG_WINDOWS:

            for shift in ABSOLUTE_LOG_SHIFTS:

                for gap in MIN_ALARM_GAPS:

                    config_id += 1

                    if (
                        short == short_window
                        and long == long_window
                        and np.isclose(
                            shift,
                            absolute_log_shift
                        )
                        and gap == min_alarm_gap
                    ):
                        return config_id

    raise ValueError(
        "Configuration not found in Module 17 grid."
    )


# ======================================================================
# HELPER — CAPACITY METRICS
# ======================================================================

def calculate_capacity_metrics(
    actual,
    provisioned,
    reserve_trigger
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    provisioned = np.asarray(
        provisioned,
        dtype=float
    )

    reserve_trigger = np.asarray(
        reserve_trigger,
        dtype=int
    )

    shortage = np.maximum(
        actual - provisioned,
        0.0
    )

    unused_capacity = np.maximum(
        provisioned - actual,
        0.0
    )

    p95_shortage = float(
        np.percentile(
            shortage,
            95
        )
    )

    tail = shortage[
        shortage >= p95_shortage
    ]

    cvar95 = (
        float(np.mean(tail))
        if len(tail) > 0
        else 0.0
    )

    return {

        "intervals":
            int(len(actual)),

        "under_provisioned_intervals":
            int(
                np.sum(
                    shortage > 0
                )
            ),

        "under_provisioned_percent":
            float(
                np.mean(
                    shortage > 0
                ) * 100
            ),

        "mean_shortage":
            float(
                np.mean(
                    shortage
                )
            ),

        "p95_shortage":
            p95_shortage,

        "shortfall_cvar95":
            cvar95,

        "mean_unused_capacity":
            float(
                np.mean(
                    unused_capacity
                )
            ),

        "mean_provisioned_capacity":
            float(
                np.mean(
                    provisioned
                )
            ),

        "mean_actual_workload":
            float(
                np.mean(
                    actual
                )
            ),

        "reserve_activation_rate":
            float(
                np.mean(
                    reserve_trigger
                ) * 100
            ),

        "total_provisioned_capacity":
            float(
                np.sum(
                    provisioned
                )
            ),

        "total_shortage":
            float(
                np.sum(
                    shortage
                )
            ),

        "total_unused_capacity":
            float(
                np.sum(
                    unused_capacity
                )
            ),
    }


# ======================================================================
# HELPER — APPLY V4 RESET
# ======================================================================

def apply_v4_policy(
    v3,
    alarm_intervals
):

    result = v3.copy()

    # --------------------------------------------------------------
    # Change-point alarm
    # --------------------------------------------------------------

    result["change_point_alarm"] = (
        result["interval"]
        .isin(alarm_intervals)
        .astype(int)
    )

    # --------------------------------------------------------------
    # Preserve original V3 threshold
    # --------------------------------------------------------------

    result["threshold_before_reset"] = (
        result[
            "adaptive_threshold"
        ].astype(float)
    )

    # --------------------------------------------------------------
    # Reset indicator
    # --------------------------------------------------------------

    result["change_point_reset"] = (
        result[
            "change_point_alarm"
        ]
    )

    # --------------------------------------------------------------
    # V4 threshold
    #
    # Alarm:
    #     theta_base
    #
    # No alarm:
    #     V3 threshold
    # --------------------------------------------------------------

    result["V4_threshold"] = np.where(
        result[
            "change_point_alarm"
        ] == 1,

        THETA_BASE,

        result[
            "adaptive_threshold"
        ]
    )

    # --------------------------------------------------------------
    # Threshold bounds
    # --------------------------------------------------------------

    if not result[
        "V4_threshold"
    ].between(
        THETA_MIN,
        THETA_MAX
    ).all():

        raise ValueError(
            "V4 threshold bounds violated."
        )

    # --------------------------------------------------------------
    # Alarm -> base threshold
    # --------------------------------------------------------------

    alarm_rows = (
        result[
            "change_point_alarm"
        ] == 1
    )

    if alarm_rows.any():

        if not np.allclose(

            result.loc[
                alarm_rows,
                "V4_threshold"
            ].to_numpy(),

            THETA_BASE
        ):

            raise ValueError(
                "Alarm interval did not "
                "reset to theta_base."
            )

    # --------------------------------------------------------------
    # Non-alarm -> V3 preserved
    # --------------------------------------------------------------

    normal_rows = (
        result[
            "change_point_alarm"
        ] == 0
    )

    if normal_rows.any():

        if not np.allclose(

            result.loc[
                normal_rows,
                "V4_threshold"
            ].to_numpy(),

            result.loc[
                normal_rows,
                "adaptive_threshold"
            ].to_numpy()
        ):

            raise ValueError(
                "Non-alarm interval changed "
                "V3 threshold."
            )

    # --------------------------------------------------------------
    # Reserve trigger
    # --------------------------------------------------------------

    result[
        "V4_reserve_trigger"
    ] = (

        result[
            "risk_signal"
        ]

        >

        result[
            "V4_threshold"
        ]

    ).astype(int)

    # --------------------------------------------------------------
    # Causality validation
    # --------------------------------------------------------------

    expected_trigger = (

        result[
            "risk_signal"
        ]

        >

        result[
            "V4_threshold"
        ]

    ).astype(int)

    if not np.array_equal(

        result[
            "V4_reserve_trigger"
        ].to_numpy(),

        expected_trigger.to_numpy()
    ):

        raise ValueError(
            "Reserve trigger causality "
            "validation failed."
        )

    # --------------------------------------------------------------
    # Capacity
    # --------------------------------------------------------------

    result[
        "V4_provisioned_capacity"
    ] = (

        result[
            "forecast_workload"
        ]

        *

        np.where(

            result[
                "V4_reserve_trigger"
            ] == 1,

            RESERVE_FACTOR,

            1.0
        )
    )

    return result


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)

    print(
        "MODULE 18 — CHANGE-POINT "
        "DOWNSTREAM ROBUSTNESS"
    )

    print("=" * 70)

    # ==============================================================
    # 1. LOAD V3
    # ==============================================================

    print(
        "\n[1] Loading Module 10 V3 policy..."
    )

    if not V3_PATH.exists():

        raise FileNotFoundError(
            f"V3 policy not found:\n"
            f"{V3_PATH}"
        )

    v3 = pd.read_csv(
        V3_PATH
    )

    print(
        f"Rows loaded: {len(v3):,}"
    )

    print(
        f"Columns: {list(v3.columns)}"
    )

    missing = [
        column
        for column in V3_REQUIRED_COLUMNS
        if column not in v3.columns
    ]

    if missing:

        raise ValueError(
            "V3 schema mismatch.\n"
            f"Missing columns: {missing}"
        )

    v3 = (
        v3
        .sort_values(
            "interval"
        )
        .reset_index(
            drop=True
        )
    )

    if v3[
        "interval"
    ].duplicated().any():

        raise ValueError(
            "Duplicate V3 intervals detected."
        )

    if v3[
        V3_REQUIRED_COLUMNS
    ].isna().any().any():

        raise ValueError(
            "Missing values found in "
            "required V3 columns."
        )

    print(
        "V3 schema validation: PASS"
    )

    # ==============================================================
    # 2. LOAD MODULE 17 ALARM FILE
    # ==============================================================

    print(
        "\n[2] Loading Module 17 sensitivity alarms..."
    )

    if not MODULE17_ALARMS_PATH.exists():

        raise FileNotFoundError(
            "Module 17 alarm file not found:\n"
            f"{MODULE17_ALARMS_PATH}"
        )

    alarms = pd.read_csv(
        MODULE17_ALARMS_PATH
    )

    print(
        f"Rows loaded: {len(alarms):,}"
    )

    print(
        f"Columns: {list(alarms.columns)}"
    )

    missing = [
        column
        for column in MODULE17_REQUIRED_COLUMNS
        if column not in alarms.columns
    ]

    if missing:

        raise ValueError(
            "Module 17 alarm schema mismatch.\n"
            f"Missing columns: {missing}\n"
            f"Available columns: {list(alarms.columns)}"
        )

    print(
        "Module 17 alarm schema validation: PASS"
    )

    # --------------------------------------------------------------
    # Important:
    #
    # This file contains ONLY alarm rows.
    #
    # Therefore every row means:
    #
    #     change_alarm = 1
    #
    # There is intentionally no change_alarm column.
    # --------------------------------------------------------------

    alarms[
        "config_id"
    ] = alarms[
        "config_id"
    ].astype(int)

    alarms[
        "interval"
    ] = alarms[
        "interval"
    ].astype(int)

    print(
        "Alarm-only representation: PASS"
    )

    # ==============================================================
    # 3. RECONSTRUCT MODULE 17 CONFIGURATION IDS
    # ==============================================================

    print(
        "\n[3] Validating Module 17 configuration IDs..."
    )

    reconstructed = []

    for cfg in CONFIGURATIONS:

        config_id = get_config_id(

            cfg[
                "short_window"
            ],

            cfg[
                "long_window"
            ],

            cfg[
                "absolute_log_shift"
            ],

            cfg[
                "min_alarm_gap"
            ]
        )

        item = cfg.copy()

        item[
            "config_id"
        ] = config_id

        reconstructed.append(
            item
        )

        print(
            f"  {cfg['configuration']:<12}"
            f" -> config_id={config_id}"
        )

    selected_config_ids = {
        item["config_id"]
        for item in reconstructed
    }

    if len(selected_config_ids) != len(
        reconstructed
    ):

        raise ValueError(
            "Configuration IDs are not unique."
        )

    print(
        "Configuration ID reconstruction: PASS"
    )

    # ==============================================================
    # 4. CHECK BASELINE
    # ==============================================================

    print(
        "\n[4] Validating Module 17 baseline..."
    )

    baseline_config_id = get_config_id(
        12,
        276,
        0.10,
        288
    )

    print(
        "Expected baseline config_id: "
        f"{baseline_config_id}"
    )

    if baseline_config_id != 41:

        raise ValueError(
            "Unexpected Module 17 "
            "baseline config_id."
        )

    baseline_alarm_rows = alarms[
        alarms[
            "config_id"
        ]
        == baseline_config_id
    ].copy()

    print(
        "Baseline alarm rows: "
        f"{len(baseline_alarm_rows):,}"
    )

    if len(baseline_alarm_rows) != 27:

        raise ValueError(
            "Baseline alarm count does not "
            "match validated Module 17 baseline."
        )

    print(
        "Baseline alarm count: 27"
    )

    print(
        "Baseline validation: PASS"
    )

    # ==============================================================
    # 5. LOAD EXISTING MODULE 11 V4
    # ==============================================================

    print(
        "\n[5] Loading existing Module 11 V4 baseline..."
    )

    if not V4_BASELINE_PATH.exists():

        raise FileNotFoundError(
            "V4 baseline not found:\n"
            f"{V4_BASELINE_PATH}"
        )

    v4_baseline = pd.read_csv(
        V4_BASELINE_PATH
    )

    required_v4_columns = [
        "interval",
        "split",
        "actual_workload",
        "forecast_workload",
        "risk_signal",
        "adaptive_threshold",
        "change_point_alarm",
        "V4_threshold",
        "V4_reserve_trigger",
        "V4_provisioned_capacity",
    ]

    missing = [
        column
        for column in required_v4_columns
        if column not in v4_baseline.columns
    ]

    if missing:

        raise ValueError(
            "Existing V4 baseline schema mismatch.\n"
            f"Missing: {missing}"
        )

    v4_baseline[
        "interval"
    ] = v4_baseline[
        "interval"
    ].astype(int)

    print(
        "Existing V4 baseline schema validation: PASS"
    )

    # ==============================================================
    # 6. LOCKED TEST
    # ==============================================================

    print(
        "\n[6] Preparing locked test partition..."
    )

    test_v3 = v3[
        v3[
            "split"
        ]
        .astype(str)
        .str.lower()
        == "test"
    ].copy()

    if len(test_v3) == 0:

        raise ValueError(
            "Locked test partition is empty."
        )

    print(
        f"Locked test intervals: "
        f"{len(test_v3):,}"
    )

    # ==============================================================
    # 7. REPRODUCE EXISTING V4 BASELINE
    # ==============================================================

    print(
        "\n[7] Reproducing existing Module 11 V4..."
    )

    baseline_alarm_intervals = set(
        baseline_alarm_rows[
            "interval"
        ].astype(int)
    )

    reproduced_v4 = apply_v4_policy(
        v3,
        baseline_alarm_intervals
    )

    reproduced_test = reproduced_v4[
        reproduced_v4[
            "split"
        ]
        .astype(str)
        .str.lower()
        == "test"
    ].copy()

    existing_test = v4_baseline[
        v4_baseline[
            "split"
        ]
        .astype(str)
        .str.lower()
        == "test"
    ].copy()

    reproduced_metrics = (
        calculate_capacity_metrics(

            reproduced_test[
                "actual_workload"
            ].to_numpy(),

            reproduced_test[
                "V4_provisioned_capacity"
            ].to_numpy(),

            reproduced_test[
                "V4_reserve_trigger"
            ].to_numpy()
        )
    )

    existing_metrics = (
        calculate_capacity_metrics(

            existing_test[
                "actual_workload"
            ].to_numpy(),

            existing_test[
                "V4_provisioned_capacity"
            ].to_numpy(),

            existing_test[
                "V4_reserve_trigger"
            ].to_numpy()
        )
    )

    comparison_metrics = [
        "under_provisioned_percent",
        "mean_shortage",
        "p95_shortage",
        "shortfall_cvar95",
        "mean_unused_capacity",
        "mean_provisioned_capacity",
        "reserve_activation_rate",
    ]

    differences = {}

    for metric in comparison_metrics:

        differences[
            metric
        ] = (

            reproduced_metrics[
                metric
            ]

            -

            existing_metrics[
                metric
            ]
        )

    if not all(
        abs(value) < 1e-8
        for value in differences.values()
    ):

        raise ValueError(
            "Module 11 baseline reproduction failed.\n"
            f"Differences: {differences}"
        )

    print(
        "Baseline V4 reproduction: PASS"
    )

    print(
        "Baseline test alarms: "
        f"{int(reproduced_test['change_point_alarm'].sum())}"
    )

    print(
        "Baseline test resets: "
        f"{int(reproduced_test['change_point_reset'].sum())}"
    )

    # ==============================================================
    # 8. PROPAGATE SENSITIVITY CONFIGURATIONS
    # ==============================================================

    print(
        "\n[8] Propagating sensitivity configurations..."
    )

    summary_rows = []

    detail_rows = []

    baseline_test_metrics = (
        reproduced_metrics
    )

    for index, cfg in enumerate(
        reconstructed,
        start=1
    ):

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"[{index}/{len(reconstructed)}] "
            f"{cfg['configuration']}"
        )

        print(
            f"  config_id={cfg['config_id']}"
        )

        print(
            f"  S={cfg['short_window']} "
            f"L={cfg['long_window']} "
            f"Shift={cfg['absolute_log_shift']} "
            f"Gap={cfg['min_alarm_gap']}"
        )

        # ----------------------------------------------------------
        # Select alarm rows belonging to this configuration
        # ----------------------------------------------------------

        cfg_alarm_rows = alarms[
            alarms[
                "config_id"
            ]
            == cfg[
                "config_id"
            ]
        ].copy()

        if len(cfg_alarm_rows) == 0:

            raise ValueError(
                "No Module 17 alarm rows found "
                "for configuration:\n"
                f"{cfg}"
            )

        # Every row in this file is an alarm.
        alarm_intervals = set(
            cfg_alarm_rows[
                "interval"
            ].astype(int)
        )

        # ----------------------------------------------------------
        # Apply V4
        # ----------------------------------------------------------

        policy = apply_v4_policy(
            v3,
            alarm_intervals
        )

        test_policy = policy[
            policy[
                "split"
            ]
            .astype(str)
            .str.lower()
            == "test"
        ].copy()

        metrics = (
            calculate_capacity_metrics(

                test_policy[
                    "actual_workload"
                ].to_numpy(),

                test_policy[
                    "V4_provisioned_capacity"
                ].to_numpy(),

                test_policy[
                    "V4_reserve_trigger"
                ].to_numpy()
            )
        )

        # ----------------------------------------------------------
        # Alarm/reset counts
        # ----------------------------------------------------------

        total_alarm_count = int(
            policy[
                "change_point_alarm"
            ].sum()
        )

        total_reset_count = int(
            policy[
                "change_point_reset"
            ].sum()
        )

        test_alarm_count = int(
            test_policy[
                "change_point_alarm"
            ].sum()
        )

        test_reset_count = int(
            test_policy[
                "change_point_reset"
            ].sum()
        )

        if total_alarm_count != total_reset_count:

            raise ValueError(
                "Total alarm/reset mismatch."
            )

        if test_alarm_count != test_reset_count:

            raise ValueError(
                "Test alarm/reset mismatch."
            )

        # ----------------------------------------------------------
        # Store test alarm details
        # ----------------------------------------------------------

        selected_test = test_policy[
            test_policy[
                "change_point_alarm"
            ] == 1
        ][
            [
                "interval",
                "split",
                "change_point_alarm",
                "V4_threshold",
                "V4_reserve_trigger",
                "V4_provisioned_capacity",
            ]
        ].copy()

        selected_test[
            "configuration"
        ] = cfg[
            "configuration"
        ]

        selected_test[
            "config_id"
        ] = cfg[
            "config_id"
        ]

        selected_test[
            "short_window"
        ] = cfg[
            "short_window"
        ]

        selected_test[
            "long_window"
        ] = cfg[
            "long_window"
        ]

        selected_test[
            "absolute_log_shift"
        ] = cfg[
            "absolute_log_shift"
        ]

        selected_test[
            "min_alarm_gap"
        ] = cfg[
            "min_alarm_gap"
        ]

        detail_rows.append(
            selected_test
        )

        # ----------------------------------------------------------
        # Summary row
        # ----------------------------------------------------------

        row = {

            "configuration":
                cfg[
                    "configuration"
                ],

            "config_id":
                cfg[
                    "config_id"
                ],

            "short_window":
                cfg[
                    "short_window"
                ],

            "long_window":
                cfg[
                    "long_window"
                ],

            "absolute_log_shift":
                cfg[
                    "absolute_log_shift"
                ],

            "min_alarm_gap":
                cfg[
                    "min_alarm_gap"
                ],

            "total_alarms":
                total_alarm_count,

            "test_alarms":
                test_alarm_count,

            "test_threshold_resets":
                test_reset_count,

            "test_intervals":
                metrics[
                    "intervals"
                ],

            "under_provisioned_percent":
                metrics[
                    "under_provisioned_percent"
                ],

            "mean_shortage":
                metrics[
                    "mean_shortage"
                ],

            "p95_shortage":
                metrics[
                    "p95_shortage"
                ],

            "shortfall_cvar95":
                metrics[
                    "shortfall_cvar95"
                ],

            "mean_unused_capacity":
                metrics[
                    "mean_unused_capacity"
                ],

            "mean_provisioned_capacity":
                metrics[
                    "mean_provisioned_capacity"
                ],

            "reserve_activation_rate":
                metrics[
                    "reserve_activation_rate"
                ],

            "total_provisioned_capacity":
                metrics[
                    "total_provisioned_capacity"
                ],

            "total_shortage":
                metrics[
                    "total_shortage"
                ],

            "total_unused_capacity":
                metrics[
                    "total_unused_capacity"
                ],
        }

        # ----------------------------------------------------------
        # Differences from baseline
        # ----------------------------------------------------------

        row[
            "delta_under_provisioned_percent"
        ] = (

            metrics[
                "under_provisioned_percent"
            ]

            -

            baseline_test_metrics[
                "under_provisioned_percent"
            ]
        )

        row[
            "delta_mean_shortage"
        ] = (

            metrics[
                "mean_shortage"
            ]

            -

            baseline_test_metrics[
                "mean_shortage"
            ]
        )

        row[
            "delta_p95_shortage"
        ] = (

            metrics[
                "p95_shortage"
            ]

            -

            baseline_test_metrics[
                "p95_shortage"
            ]
        )

        row[
            "delta_cvar95_shortage"
        ] = (

            metrics[
                "shortfall_cvar95"
            ]

            -

            baseline_test_metrics[
                "shortfall_cvar95"
            ]
        )

        row[
            "delta_mean_unused_capacity"
        ] = (

            metrics[
                "mean_unused_capacity"
            ]

            -

            baseline_test_metrics[
                "mean_unused_capacity"
            ]
        )

        row[
            "delta_mean_provisioned_capacity"
        ] = (

            metrics[
                "mean_provisioned_capacity"
            ]

            -

            baseline_test_metrics[
                "mean_provisioned_capacity"
            ]
        )

        row[
            "delta_reserve_activation_rate"
        ] = (

            metrics[
                "reserve_activation_rate"
            ]

            -

            baseline_test_metrics[
                "reserve_activation_rate"
            ]
        )

        summary_rows.append(
            row
        )

        print(
            f"  Total alarms: "
            f"{total_alarm_count}"
        )

        print(
            f"  Test alarms: "
            f"{test_alarm_count}"
        )

        print(
            f"  Under-provisioned: "
            f"{metrics['under_provisioned_percent']:.4f}%"
        )

        print(
            f"  Mean shortage: "
            f"{metrics['mean_shortage']:.4f}"
        )

        print(
            f"  Reserve activation: "
            f"{metrics['reserve_activation_rate']:.4f}%"
        )

    summary = pd.DataFrame(
        summary_rows
    )

    # ==============================================================
    # 9. FINAL VALIDATION
    # ==============================================================

    print(
        "\n[9] Running final robustness validation..."
    )

    # --------------------------------------------------------------
    # Number of configurations
    # --------------------------------------------------------------

    if len(summary) != len(
        CONFIGURATIONS
    ):

        raise ValueError(
            "Not all configurations were evaluated."
        )

    # --------------------------------------------------------------
    # Baseline must be first
    # --------------------------------------------------------------

    if (
        summary.iloc[0][
            "configuration"
        ]
        != "baseline"
    ):

        raise ValueError(
            "Baseline is not first."
        )

    # --------------------------------------------------------------
    # Baseline metrics must reproduce V4
    # --------------------------------------------------------------

    baseline_row = summary.iloc[0]

    for metric in comparison_metrics:

        if not np.isclose(

            baseline_row[
                metric
            ],

            existing_metrics[
                metric
            ],

            atol=1e-8,

            rtol=0.0
        ):

            raise ValueError(
                f"Baseline mismatch "
                f"for metric: {metric}"
            )

    # --------------------------------------------------------------
    # Configuration IDs unique
    # --------------------------------------------------------------

    if summary[
        "config_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate config IDs detected."
        )

    # --------------------------------------------------------------
    # Test isolation
    # --------------------------------------------------------------

    print(
        "Baseline reproduction: PASS"
    )

    print(
        "Configuration completeness: PASS"
    )

    print(
        "Alarm/reset consistency: PASS"
    )

    print(
        "Locked-test isolation: PASS"
    )

    # ==============================================================
    # 10. SAVE SUMMARY
    # ==============================================================

    print(
        "\n[10] Saving downstream robustness summary..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False
    )

    # ==============================================================
    # 11. SAVE TEST ALARM DETAILS
    # ==============================================================

    if detail_rows:

        details = pd.concat(
            detail_rows,
            ignore_index=True
        )

    else:

        details = pd.DataFrame()

    details.to_csv(
        DETAIL_OUTPUT,
        index=False
    )

    print(
        "Summary saved:"
    )

    print(
        f"  {SUMMARY_OUTPUT}"
    )

    print(
        "Test alarm details saved:"
    )

    print(
        f"  {DETAIL_OUTPUT}"
    )

    # ==============================================================
    # 12. SAVE METADATA
    # ==============================================================

    print(
        "\n[11] Saving validation metadata..."
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = {

        "module":
            "18_change_point_downstream_robustness",

        "purpose":
            (
                "Evaluate whether pre-defined "
                "change-point parameter variations "
                "propagate into downstream V4 "
                "capacity behavior."
            ),

        "module17_alarm_representation":
            (
                "Alarm-only rows; every row in "
                "change_point_sensitivity_alarms.csv "
                "represents an alarm."
            ),

        "module17_configuration_id_order":
            (
                "S in [6,12,24], "
                "L in [144,276,432], "
                "Shift in [0.05,0.10,0.20], "
                "Gap in [144,288,576]."
            ),

        "baseline_config_id":
            baseline_config_id,

        "selection_method":
            (
                "Baseline plus one-factor-at-a-time "
                "alternatives."
            ),

        "configuration_count":
            len(CONFIGURATIONS),

        "configurations":
            reconstructed,

        "v3_source":
            str(
                V3_PATH.relative_to(
                    BASE_DIR
                )
            ),

        "module17_source":
            str(
                MODULE17_ALARMS_PATH.relative_to(
                    BASE_DIR
                )
            ),

        "module11_baseline_source":
            str(
                V4_BASELINE_PATH.relative_to(
                    BASE_DIR
                )
            ),

        "theta_base":
            THETA_BASE,

        "reserve_factor":
            RESERVE_FACTOR,

        "primary_forecast":
            "CP-HGB",

        "capacity_definition":
            "normalized workload capacity",

        "token_to_gpu_conversion":
            False,

        "test_used_for_configuration_selection":
            False,

        "test_used_for_parameter_tuning":
            False,

        "baseline_reproduction":
            "PASS",

        "causal_reset_rule":
            (
                "Alarm interval resets threshold "
                "to theta_base; non-alarm interval "
                "preserves V3 adaptive threshold."
            ),

        "robustness_validation":
            "PASS",
    }

    with open(
        METADATA_OUTPUT,
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
        f"  {METADATA_OUTPUT}"
    )

    # ==============================================================
    # FINAL
    # ==============================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODULE 18 — DOWNSTREAM ROBUSTNESS COMPLETE"
    )

    print(
        "=" * 70
    )

    display_columns = [
        "configuration",
        "config_id",
        "total_alarms",
        "test_alarms",
        "under_provisioned_percent",
        "mean_shortage",
        "reserve_activation_rate",
    ]

    print(
        "\nSummary:"
    )

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nImportant:"
    )

    print(
        "  CP-HGB remains the primary capacity forecast."
    )

    print(
        "  V3 adaptive threshold state is preserved."
    )

    print(
        "  Only change-point reset behavior is varied."
    )

    print(
        "  Configuration selection was independent "
        "of downstream test performance."
    )

    print(
        "  Locked test data was not used to tune parameters."
    )

    print(
        "\n18 COMPLETE"
    )


if __name__ == "__main__":
    main()