import json
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# PATHS
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHANGE_POINT_PATH = (
    BASE_DIR
    / "results"
    / "alarms"
    / "change_point_5min.csv"
)

V3_PATH = (
    BASE_DIR
    / "results"
    / "policies"
    / "V3_adaptive_risk.csv"
)

ADAPTIVE_METADATA_PATH = (
    BASE_DIR
    / "results"
    / "validation"
    / "adaptive_threshold_metadata.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "policies"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "V4_adaptive_reset.csv"
)

VALIDATION_DIR = (
    BASE_DIR
    / "results"
    / "validation"
)

METADATA_PATH = (
    VALIDATION_DIR
    / "change_point_reset_metadata.json"
)


# ======================================================================
# PARAMETERS
# ======================================================================

# Paper 4 validation-selected base threshold
THETA_BASE = 0.9591

# Same reserve factor used by V2/V3
RESERVE_FACTOR = 1.10

# Same adaptive threshold parameters used in Module 09
ETA = 0.01
THETA_MIN = 0.50
THETA_MAX = 0.99

# Historical window used by Module 09
HISTORY_WINDOW = 60


# ======================================================================
# EXPECTED V3 SCHEMA
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
# HELPERS
# ======================================================================

def calculate_capacity_metrics(
    actual,
    provisioned,
    reserve_trigger
):
    """
    Calculate normalized capacity metrics.
    """

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
            int(np.sum(shortage > 0)),

        "under_provisioned_percent":
            float(
                np.mean(shortage > 0) * 100
            ),

        "mean_shortage":
            float(np.mean(shortage)),

        "p95_shortage":
            p95_shortage,

        "shortfall_cvar95":
            cvar95,

        "mean_unused_capacity":
            float(
                np.mean(unused_capacity)
            ),

        "mean_provisioned_capacity":
            float(
                np.mean(provisioned)
            ),

        "mean_actual_workload":
            float(
                np.mean(actual)
            ),

        "reserve_activation_rate":
            float(
                np.mean(reserve_trigger) * 100
            ),

        "total_provisioned_capacity":
            float(
                np.sum(provisioned)
            ),

        "total_shortage":
            float(
                np.sum(shortage)
            ),

        "total_unused_capacity":
            float(
                np.sum(unused_capacity)
            )
    }


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("PAPER 3 CHANGE-POINT RESET — V4")
    print("=" * 70)

    # ==================================================================
    # 1. LOAD V3
    # ==================================================================

    print("\n[1] Loading V3 adaptive-risk policy...")

    if not V3_PATH.exists():
        raise FileNotFoundError(
            f"V3 policy file not found:\n{V3_PATH}"
        )

    v3 = pd.read_csv(V3_PATH)

    print(
        f"Rows loaded: {len(v3):,}"
    )

    print(
        f"Columns: {list(v3.columns)}"
    )

    # ==================================================================
    # 2. VALIDATE V3
    # ==================================================================

    print("\n[2] Validating V3 schema...")

    missing = [
        col
        for col in V3_REQUIRED_COLUMNS
        if col not in v3.columns
    ]

    if missing:
        raise ValueError(
            "Missing V3 columns:\n"
            + "\n".join(
                f"  - {col}"
                for col in missing
            )
        )

    print(
        "V3 schema validation: PASS"
    )

    # ==================================================================
    # 3. LOAD CHANGE-POINT OUTPUT
    # ==================================================================

    print(
        "\n[3] Loading Paper 3 change-point alarms..."
    )

    if not CHANGE_POINT_PATH.exists():
        raise FileNotFoundError(
            f"Change-point file not found:\n"
            f"{CHANGE_POINT_PATH}"
        )

    cp = pd.read_csv(
        CHANGE_POINT_PATH
    )

    print(
        f"Rows loaded: {len(cp):,}"
    )

    print(
        f"Columns: {list(cp.columns)}"
    )

    # ==================================================================
    # 4. USE ACTUAL MODULE 03 SCHEMA
    # ==================================================================

    print(
        "\n[4] Validating Paper 3 change-point schema..."
    )

    required_cp_columns = [
        "interval",
        "change_alarm",
        "change_score",
        "short_long_ratio",
        "elapsed_bins_since_alarm"
    ]

    missing_cp = [
        col
        for col in required_cp_columns
        if col not in cp.columns
    ]

    if missing_cp:
        raise ValueError(
            "Missing Module 03 columns:\n"
            + "\n".join(
                f"  - {col}"
                for col in missing_cp
            )
        )

    print(
        "Module 03 schema validation: PASS"
    )

    print(
        "Change-point alarm column: change_alarm"
    )

    # ==================================================================
    # 5. VALIDATE CHANGE-POINT DATA
    # ==================================================================

    print(
        "\n[5] Validating change-point alarms..."
    )

    cp = (
        cp
        .sort_values("interval")
        .reset_index(drop=True)
    )

    if cp["interval"].duplicated().any():

        raise ValueError(
            "Duplicate intervals detected in "
            "change-point output."
        )

    cp["change_alarm"] = (
        cp["change_alarm"]
        .astype(int)
    )

    alarm_values = sorted(
        cp["change_alarm"]
        .unique()
        .tolist()
    )

    if not set(alarm_values).issubset({0, 1}):

        raise ValueError(
            "change_alarm must contain only 0 and 1."
        )

    total_alarms = int(
        cp["change_alarm"].sum()
    )

    print(
        f"Alarm values: {alarm_values}"
    )

    print(
        f"Total alarms: {total_alarms}"
    )

    print(
        "Change-point validation: PASS"
    )

    # ==================================================================
    # 6. NORMALIZE INTERVAL
    # ==================================================================

    print(
        "\n[6] Normalizing interval identifiers..."
    )

    v3["interval"] = (
        v3["interval"]
        .astype(int)
    )

    cp["interval"] = (
        cp["interval"]
        .astype(int)
    )

    # ==================================================================
    # 7. ALIGN CHANGE-POINT ALARMS
    # ==================================================================

    print(
        "\n[7] Aligning change-point alarms with V3..."
    )

    cp_alignment = cp[
        [
            "interval",
            "change_alarm",
            "change_score",
            "short_long_ratio",
            "elapsed_bins_since_alarm"
        ]
    ].copy()

    cp_alignment = cp_alignment.rename(
        columns={
            "change_alarm":
                "change_point_alarm"
        }
    )

    v4 = v3.merge(
        cp_alignment,
        on="interval",
        how="left",
        validate="one_to_one"
    )

    # No alarm is represented as 0.
    v4["change_point_alarm"] = (
        v4["change_point_alarm"]
        .fillna(0)
        .astype(int)
    )

    print(
        f"V3 intervals: {len(v3):,}"
    )

    print(
        f"V4 intervals: {len(v4):,}"
    )

    print(
        f"Aligned alarms: "
        f"{int(v4['change_point_alarm'].sum()):,}"
    )

    # ==================================================================
    # 8. LOAD MODULE 09 METADATA
    # ==================================================================

    print(
        "\n[8] Loading adaptive-threshold metadata..."
    )

    if not ADAPTIVE_METADATA_PATH.exists():

        raise FileNotFoundError(
            "Module 09 metadata not found:\n"
            f"{ADAPTIVE_METADATA_PATH}"
        )

    with open(
        ADAPTIVE_METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        adaptive_metadata = json.load(
            file
        )

    if (
        "target_failure_rate"
        not in adaptive_metadata
    ):

        raise ValueError(
            "Module 09 metadata does not contain "
            "'target_failure_rate'."
        )

    target_failure_rate = float(
        adaptive_metadata[
            "target_failure_rate"
        ]
    )

    print(
        f"Target failure rate: "
        f"{target_failure_rate:.6f}"
    )

    print(
        "Source: Module 09 calibration"
    )

    # ==================================================================
    # 9. IMPORTANT DESIGN:
    #
    # V3 ALREADY CONTAINS THE ADAPTIVE THRESHOLD STATE.
    #
    # V4 should preserve V3's adaptive state except when a change-point
    # alarm occurs.
    #
    # Therefore:
    #
    #     normal interval:
    #         V4 threshold = V3 adaptive threshold
    #
    #     alarm interval:
    #         V4 threshold = theta_base
    #
    # This isolates the contribution of the reset mechanism.
    #
    # ==================================================================

    print(
        "\n[9] Applying causal change-point reset..."
    )

    v4 = (
        v4
        .sort_values("interval")
        .reset_index(drop=True)
    )

    v4["threshold_before_reset"] = (
        v4["adaptive_threshold"]
        .astype(float)
    )

    v4["change_point_reset"] = (
        v4["change_point_alarm"]
        .astype(int)
    )

    # Reset threshold only on detected change-point intervals.
    v4["V4_threshold"] = np.where(
        v4["change_point_alarm"] == 1,
        THETA_BASE,
        v4["adaptive_threshold"]
    )

    # ==================================================================
    # 10. RESET CAUSALITY CHECK
    # ==================================================================

    print(
        "\n[10] Checking reset causality..."
    )

    # All thresholds must be within the configured bounds.
    if not (
        v4["V4_threshold"]
        .between(
            THETA_MIN,
            THETA_MAX
        )
        .all()
    ):

        raise ValueError(
            "V4 threshold bounds violated."
        )

    # Every alarm interval must reset to theta_base.
    alarm_rows = (
        v4["change_point_alarm"] == 1
    )

    if alarm_rows.any():

        reset_values = (
            v4.loc[
                alarm_rows,
                "V4_threshold"
            ]
        )

        if not np.allclose(
            reset_values.to_numpy(),
            THETA_BASE
        ):

            raise ValueError(
                "Change-point reset did not "
                "set threshold to theta_base."
            )

    # Every non-alarm interval must preserve V3 state.
    normal_rows = (
        v4["change_point_alarm"] == 0
    )

    if normal_rows.any():

        normal_v3 = (
            v4.loc[
                normal_rows,
                "adaptive_threshold"
            ]
            .to_numpy()
        )

        normal_v4 = (
            v4.loc[
                normal_rows,
                "V4_threshold"
            ]
            .to_numpy()
        )

        if not np.allclose(
            normal_v3,
            normal_v4
        ):

            raise ValueError(
                "V4 changed the adaptive threshold "
                "outside change-point intervals."
            )

    print(
        "Threshold bounds: PASS"
    )

    print(
        "Alarm -> base-threshold reset: PASS"
    )

    print(
        "Non-alarm -> V3 threshold preserved: PASS"
    )

    print(
        "Reset causality: PASS"
    )

    # ==================================================================
    # 11. CALCULATE V4 RESERVE TRIGGER
    # ==================================================================

    print(
        "\n[11] Calculating V4 reserve decisions..."
    )

    v4["V4_reserve_trigger"] = (
        v4["risk_signal"]
        >
        v4["V4_threshold"]
    ).astype(int)

    # ==================================================================
    # 12. CALCULATE V4 PROVISIONED CAPACITY
    # ==================================================================

    v4["V4_provisioned_capacity"] = (
        v4["forecast_workload"]
        *
        np.where(
            v4["V4_reserve_trigger"] == 1,
            RESERVE_FACTOR,
            1.0
        )
    )

    # ==================================================================
    # 13. RESERVE DECISION CHECK
    # ==================================================================

    print(
        "\n[12] Validating reserve decisions..."
    )

    expected_trigger = (
        v4["risk_signal"]
        >
        v4["V4_threshold"]
    ).astype(int)

    if not np.array_equal(
        expected_trigger.to_numpy(),
        v4[
            "V4_reserve_trigger"
        ].to_numpy()
    ):

        raise ValueError(
            "V4 reserve trigger validation failed."
        )

    print(
        "Reserve trigger causality: PASS"
    )

    # ==================================================================
    # 14. CAPACITY METRICS
    # ==================================================================

    print(
        "\n[13] Calculating V4 capacity metrics..."
    )

    actual = (
        v4["actual_workload"]
        .astype(float)
        .to_numpy()
    )

    provisioned = (
        v4["V4_provisioned_capacity"]
        .astype(float)
        .to_numpy()
    )

    reserve_trigger = (
        v4["V4_reserve_trigger"]
        .astype(int)
        .to_numpy()
    )

    metrics = calculate_capacity_metrics(
        actual,
        provisioned,
        reserve_trigger
    )

    print(
        "\nV4:"
    )

    print(
        f" Under-provisioned: "
        f"{metrics['under_provisioned_percent']:.4f}%"
    )

    print(
        f" Mean shortage: "
        f"{metrics['mean_shortage']:.2f}"
    )

    print(
        f" P95 shortage: "
        f"{metrics['p95_shortage']:.2f}"
    )

    print(
        f" CVaR95 shortage: "
        f"{metrics['shortfall_cvar95']:.2f}"
    )

    print(
        f" Mean unused capacity: "
        f"{metrics['mean_unused_capacity']:.2f}"
    )

    print(
        f" Mean provisioned capacity: "
        f"{metrics['mean_provisioned_capacity']:.2f}"
    )

    print(
        f" Reserve activation: "
        f"{metrics['reserve_activation_rate']:.4f}%"
    )

    # ==================================================================
    # 15. RESET SUMMARY
    # ==================================================================

    print(
        "\n[14] Change-point reset summary..."
    )

    reset_count = int(
        v4["change_point_reset"].sum()
    )

    print(
        f"Change-point alarms: {total_alarms}"
    )

    print(
        f"Threshold resets: {reset_count}"
    )

    if reset_count != total_alarms:

        raise ValueError(
            "Number of resets does not equal "
            "number of change-point alarms."
        )

    # ==================================================================
    # 16. SAVE V4
    # ==================================================================

    print(
        "\n[15] Saving V4 policy..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_columns = [
        "interval",
        "split",
        "actual_next_5min_tokens",
        "hgb_prediction",
        "cp_hgb_prediction",
        "actual_workload",
        "forecast_workload",
        "risk_signal",
        "adaptive_threshold",
        "change_point_alarm",
        "change_score",
        "short_long_ratio",
        "elapsed_bins_since_alarm",
        "change_point_reset",
        "threshold_before_reset",
        "V4_threshold",
        "V4_reserve_trigger",
        "V4_provisioned_capacity"
    ]

    v4[
        output_columns
    ].to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"V4 saved:\n"
        f"  {OUTPUT_PATH}"
    )

    # ==================================================================
    # 17. SAVE METADATA
    # ==================================================================

    print(
        "\n[16] Saving validation metadata..."
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = {

        "module":
            "11_change_point_reset",

        "method":
            "adaptive_threshold_with_change_point_reset",

        "change_point_source":
            str(
                CHANGE_POINT_PATH.relative_to(
                    BASE_DIR
                )
            ),

        "V3_source":
            str(
                V3_PATH.relative_to(
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

        "reserve_factor":
            RESERVE_FACTOR,

        "target_failure_rate":
            target_failure_rate,

        "change_point_alarms":
            total_alarms,

        "threshold_resets":
            reset_count,

        "simulation_intervals":
            int(len(v4)),

        "V4_metrics":
            metrics,

        "reset_rule":
            "On change-point alarm, reset adaptive risk threshold to theta_base before the capacity decision.",

        "non_alarm_rule":
            "When no change-point alarm occurs, preserve the V3 adaptive threshold.",

        "test_parameter_tuning":
            False,

        "test_outcomes_used_for_reset":
            False,

        "token_to_gpu_conversion":
            False,

        "production_autoscaling_claim":
            False,

        "causal_validation":
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
        f"Metadata saved:\n"
        f"  {METADATA_PATH}"
    )

    # ==================================================================
    # FINAL
    # ==================================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "V4 CHANGE-POINT RESET COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nCreated:"
    )

    print(
        "  results/policies/"
        "V4_adaptive_reset.csv"
    )

    print(
        "  results/validation/"
        "change_point_reset_metadata.json"
    )

    print(
        "\nV4 definition:"
    )

    print(
        "  CP-HGB forecast"
    )

    print(
        "  + adaptive failure-risk threshold"
    )

    print(
        "  + 10% reserve"
    )

    print(
        "  + Paper 3 change-point reset"
    )

    print(
        "\n11 COMPLETE"
    )


if __name__ == "__main__":
    main()