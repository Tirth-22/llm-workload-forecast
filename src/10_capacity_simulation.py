import json
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

WORKLOAD_PATH = (
    BASE_DIR
    / "results"
    / "predictions"
    / "workload_forecast_predictions.csv"
)

RISK_PATH = (
    BASE_DIR
    / "results"
    / "predictions"
    / "calibrated_failure_risk_predictions.csv"
)

ADAPTIVE_THRESHOLD_PATH = (
    BASE_DIR
    / "results"
    / "policies"
    / "adaptive_threshold_predictions.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "policies"
)

VALIDATION_DIR = (
    BASE_DIR
    / "results"
    / "validation"
)

V1_OUTPUT = OUTPUT_DIR / "V1_forecast_only.csv"
V2_OUTPUT = OUTPUT_DIR / "V2_fixed_risk.csv"
V3_OUTPUT = OUTPUT_DIR / "V3_adaptive_risk.csv"

METADATA_OUTPUT = (
    VALIDATION_DIR
    / "capacity_simulation_metadata.json"
)


# ======================================================================
# FIXED PARAMETERS
# ======================================================================

FIXED_THRESHOLD = 0.9591

RESERVE_FACTOR = 1.10

MINUTES_PER_INTERVAL = 5


# ======================================================================
# EXPECTED MODULE 05 COLUMNS
# ======================================================================

INTERVAL_COLUMN = "interval"

ACTUAL_COLUMN = "actual_next_5min_tokens"

HGB_COLUMN = "hgb_prediction"

CP_HGB_COLUMN = "cp_hgb_prediction"


# ======================================================================
# EXPECTED MODULE 08 COLUMNS
# ======================================================================

RISK_MINUTE_COLUMN = "minute"

RISK_PROBABILITY_COLUMN = (
    "calibrated_failure_probability"
)


# ======================================================================
# EXPECTED MODULE 09 COLUMNS
# ======================================================================

ADAPTIVE_THRESHOLD_COLUMN = (
    "threshold_before_update"
)


# ======================================================================
# CAPACITY METRICS
# ======================================================================

def calculate_capacity_metrics(
    actual,
    provisioned,
    reserve_trigger
):
    """
    Calculate normalized workload-capacity metrics.
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

    if len(shortage) > 0:

        p95_shortage = float(
            np.percentile(
                shortage,
                95
            )
        )

        cvar_threshold = p95_shortage

        tail = shortage[
            shortage >= cvar_threshold
        ]

        if len(tail) > 0:
            shortfall_cvar95 = float(
                np.mean(tail)
            )
        else:
            shortfall_cvar95 = 0.0

    else:

        p95_shortage = 0.0
        shortfall_cvar95 = 0.0

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
                np.mean(shortage)
            ),

        "p95_shortage":
            p95_shortage,

        "shortfall_cvar95":
            shortfall_cvar95,

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
                np.mean(actual)
            ),

        "reserve_activation_rate":
            float(
                np.mean(
                    reserve_trigger
                ) * 100
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
    print("FORECAST-DRIVEN CAPACITY SIMULATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. LOAD PAPER 3 WORKLOAD PREDICTIONS
    # ------------------------------------------------------------------

    print("\n[1] Loading Paper 3 workload forecasts...")

    if not WORKLOAD_PATH.exists():
        raise FileNotFoundError(
            f"Workload prediction file not found:\n"
            f"{WORKLOAD_PATH}"
        )

    workload = pd.read_csv(
        WORKLOAD_PATH
    )

    print(
        f"Rows loaded: {len(workload):,}"
    )

    print(
        f"Columns: {list(workload.columns)}"
    )

    # ------------------------------------------------------------------
    # 2. VALIDATE EXACT MODULE 05 SCHEMA
    # ------------------------------------------------------------------

    print("\n[2] Validating Module 05 schema...")

    required_workload_columns = [
        INTERVAL_COLUMN,
        "split",
        ACTUAL_COLUMN,
        HGB_COLUMN,
        CP_HGB_COLUMN
    ]

    missing = [
        column
        for column in required_workload_columns
        if column not in workload.columns
    ]

    if missing:
        raise ValueError(
            "Module 05 schema mismatch.\n"
            f"Missing columns: {missing}\n"
            f"Available columns: {list(workload.columns)}"
        )

    print(
        f"Interval column : {INTERVAL_COLUMN}"
    )

    print(
        f"Actual workload : {ACTUAL_COLUMN}"
    )

    print(
        f"HGB forecast    : {HGB_COLUMN}"
    )

    print(
        f"CP-HGB forecast : {CP_HGB_COLUMN}"
    )

    print(
        "Module 05 schema validation: PASS"
    )

    # ------------------------------------------------------------------
    # 3. VALIDATE WORKLOAD DATA
    # ------------------------------------------------------------------

    print("\n[3] Validating workload data...")

    workload = (
        workload
        .sort_values(INTERVAL_COLUMN)
        .reset_index(drop=True)
    )

    if workload[
        INTERVAL_COLUMN
    ].duplicated().any():

        raise ValueError(
            "Duplicate workload intervals detected."
        )

    required_numeric = [
        ACTUAL_COLUMN,
        HGB_COLUMN,
        CP_HGB_COLUMN
    ]

    if workload[
        required_numeric
    ].isna().any().any():

        print(
            workload[
                required_numeric
            ]
            .isna()
            .sum()
        )

        raise ValueError(
            "Missing values found in workload prediction data."
        )

    for column in required_numeric:

        if (
            workload[column] < 0
        ).any():

            raise ValueError(
                f"Negative value found in {column}."
            )

    print(
        "Missing values: 0"
    )

    print(
        "Duplicate intervals: 0"
    )

    print(
        "Non-negative workload values: PASS"
    )

    # ------------------------------------------------------------------
    # 4. LOAD CALIBRATED RISK
    # ------------------------------------------------------------------

    print("\n[4] Loading calibrated failure-risk predictions...")

    if not RISK_PATH.exists():
        raise FileNotFoundError(
            f"Risk prediction file not found:\n"
            f"{RISK_PATH}"
        )

    risk = pd.read_csv(
        RISK_PATH
    )

    print(
        f"Rows loaded: {len(risk):,}"
    )

    required_risk_columns = [
        RISK_MINUTE_COLUMN,
        "split",
        "failure",
        RISK_PROBABILITY_COLUMN
    ]

    missing = [
        column
        for column in required_risk_columns
        if column not in risk.columns
    ]

    if missing:
        raise ValueError(
            "Module 08 schema mismatch.\n"
            f"Missing columns: {missing}\n"
            f"Available columns: {list(risk.columns)}"
        )

    print(
        "Risk schema validation: PASS"
    )

    # ------------------------------------------------------------------
    # 5. BUILD FIVE-MINUTE RISK VIEW
    # ------------------------------------------------------------------

    print(
        "\n[5] Aggregating minute-level risk to five-minute intervals..."
    )

    risk[
        RISK_MINUTE_COLUMN
    ] = (
        risk[
            RISK_MINUTE_COLUMN
        ]
        .astype(int)
    )

    risk[
        "interval"
    ] = (
        risk[
            RISK_MINUTE_COLUMN
        ]
        // MINUTES_PER_INTERVAL
    )

    # Conservative risk aggregation:
    # maximum calibrated risk within the interval.

    risk_5min = (
        risk
        .groupby(
            "interval",
            as_index=False
        )
        .agg(
            risk_signal=(
                RISK_PROBABILITY_COLUMN,
                "max"
            )
        )
    )

    print(
        f"Five-minute risk intervals: "
        f"{len(risk_5min):,}"
    )

    # ------------------------------------------------------------------
    # 6. LOAD MODULE 09 ADAPTIVE THRESHOLDS
    # ------------------------------------------------------------------

    print(
        "\n[6] Loading adaptive threshold predictions..."
    )

    if not ADAPTIVE_THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Adaptive threshold file not found:\n"
            f"{ADAPTIVE_THRESHOLD_PATH}"
        )

    adaptive = pd.read_csv(
        ADAPTIVE_THRESHOLD_PATH
    )

    print(
        f"Rows loaded: {len(adaptive):,}"
    )

    required_adaptive_columns = [
        "minute",
        "split",
        ADAPTIVE_THRESHOLD_COLUMN
    ]

    missing = [
        column
        for column in required_adaptive_columns
        if column not in adaptive.columns
    ]

    if missing:
        raise ValueError(
            "Module 09 schema mismatch.\n"
            f"Missing columns: {missing}\n"
            f"Available columns: {list(adaptive.columns)}"
        )

    print(
        "Adaptive threshold schema validation: PASS"
    )

    # ------------------------------------------------------------------
    # 7. CREATE FIVE-MINUTE ADAPTIVE THRESHOLD
    # ------------------------------------------------------------------

    print(
        "\n[7] Building five-minute adaptive threshold view..."
    )

    adaptive[
        "interval"
    ] = (
        adaptive[
            "minute"
        ]
        .astype(int)
        // MINUTES_PER_INTERVAL
    )

    adaptive = (
        adaptive
        .sort_values("minute")
        .reset_index(drop=True)
    )

    # First threshold available in each interval.
    #
    # This represents the threshold at the start of the interval.
    # Later observations in the same interval cannot change the
    # already-made capacity decision.

    adaptive_5min = (
        adaptive
        .groupby(
            "interval",
            as_index=False
        )
        .first()
    )

    adaptive_5min = adaptive_5min[
        [
            "interval",
            ADAPTIVE_THRESHOLD_COLUMN
        ]
    ]

    adaptive_5min = adaptive_5min.rename(
        columns={
            ADAPTIVE_THRESHOLD_COLUMN:
                "adaptive_threshold"
        }
    )

    print(
        f"Adaptive threshold intervals: "
        f"{len(adaptive_5min):,}"
    )

    # ------------------------------------------------------------------
    # 8. MERGE ALL COMPONENTS
    # ------------------------------------------------------------------

    print(
        "\n[8] Combining workload, risk, and adaptive threshold..."
    )

    simulation = workload.merge(
        risk_5min,
        on="interval",
        how="inner"
    )

    simulation = simulation.merge(
        adaptive_5min,
        on="interval",
        how="left"
    )

    print(
        f"Rows after workload/risk merge: "
        f"{len(simulation):,}"
    )

    # ------------------------------------------------------------------
    # 9. VALIDATE MERGED DATA
    # ------------------------------------------------------------------

    print(
        "\n[9] Validating simulation matrix..."
    )

    required_simulation_columns = [
        ACTUAL_COLUMN,
        HGB_COLUMN,
        CP_HGB_COLUMN,
        "risk_signal",
        "adaptive_threshold"
    ]

    missing_values = (
        simulation[
            required_simulation_columns
        ]
        .isna()
        .sum()
    )

    if missing_values.any():

        print(
            "Missing merged values:"
        )

        print(
            missing_values[
                missing_values > 0
            ]
        )

        raise ValueError(
            "Simulation matrix contains missing values."
        )

    if (
        simulation[
            "risk_signal"
        ]
        < 0
    ).any() or (
        simulation[
            "risk_signal"
        ]
        > 1
    ).any():

        raise ValueError(
            "Risk probability outside [0,1]."
        )

    print(
        "Missing values after merge: 0"
    )

    print(
        "Risk probability validation: PASS"
    )

    print(
        "Simulation matrix validation: PASS"
    )

    # ------------------------------------------------------------------
    # 10. PREPARE PRIMARY FORECAST
    # ------------------------------------------------------------------

    print(
        "\n[10] Preparing CP-HGB forecast as primary capacity signal..."
    )

    simulation[
        "actual_workload"
    ] = (
        simulation[
            ACTUAL_COLUMN
        ]
        .astype(float)
        .clip(lower=0)
    )

    simulation[
        "forecast_workload"
    ] = (
        simulation[
            CP_HGB_COLUMN
        ]
        .astype(float)
        .clip(lower=0)
    )

    print(
        "Primary forecast: CP-HGB"
    )

    # ------------------------------------------------------------------
    # 11. V1 — FORECAST ONLY
    # ------------------------------------------------------------------

    print(
        "\n[11] Simulating V1 — forecast only..."
    )

    simulation[
        "V1_reserve_trigger"
    ] = 0

    simulation[
        "V1_provisioned_capacity"
    ] = (
        simulation[
            "forecast_workload"
        ]
    )

    # ------------------------------------------------------------------
    # 12. V2 — FIXED RISK
    # ------------------------------------------------------------------

    print(
        "\n[12] Simulating V2 — fixed risk threshold..."
    )

    simulation[
        "V2_reserve_trigger"
    ] = (
        simulation[
            "risk_signal"
        ]
        >
        FIXED_THRESHOLD
    ).astype(int)

    simulation[
        "V2_provisioned_capacity"
    ] = (
        simulation[
            "forecast_workload"
        ]
        *
        np.where(
            simulation[
                "V2_reserve_trigger"
            ] == 1,
            RESERVE_FACTOR,
            1.0
        )
    )

    # ------------------------------------------------------------------
    # 13. V3 — ADAPTIVE RISK
    # ------------------------------------------------------------------

    print(
        "\n[13] Simulating V3 — adaptive risk threshold..."
    )

    simulation[
        "V3_reserve_trigger"
    ] = (
        simulation[
            "risk_signal"
        ]
        >
        simulation[
            "adaptive_threshold"
        ]
    ).astype(int)

    simulation[
        "V3_provisioned_capacity"
    ] = (
        simulation[
            "forecast_workload"
        ]
        *
        np.where(
            simulation[
                "V3_reserve_trigger"
            ] == 1,
            RESERVE_FACTOR,
            1.0
        )
    )

    # ------------------------------------------------------------------
    # 14. CALCULATE METRICS
    # ------------------------------------------------------------------

    print(
        "\n[14] Calculating policy metrics..."
    )

    actual = (
        simulation[
            "actual_workload"
        ]
        .to_numpy()
    )

    v1_capacity = (
        simulation[
            "V1_provisioned_capacity"
        ]
        .to_numpy()
    )

    v2_capacity = (
        simulation[
            "V2_provisioned_capacity"
        ]
        .to_numpy()
    )

    v3_capacity = (
        simulation[
            "V3_provisioned_capacity"
        ]
        .to_numpy()
    )

    v1_trigger = (
        simulation[
            "V1_reserve_trigger"
        ]
        .to_numpy()
    )

    v2_trigger = (
        simulation[
            "V2_reserve_trigger"
        ]
        .to_numpy()
    )

    v3_trigger = (
        simulation[
            "V3_reserve_trigger"
        ]
        .to_numpy()
    )

    v1_metrics = calculate_capacity_metrics(
        actual,
        v1_capacity,
        v1_trigger
    )

    v2_metrics = calculate_capacity_metrics(
        actual,
        v2_capacity,
        v2_trigger
    )

    v3_metrics = calculate_capacity_metrics(
        actual,
        v3_capacity,
        v3_trigger
    )

    # ------------------------------------------------------------------
    # 15. DISPLAY RESULTS
    # ------------------------------------------------------------------

    print("\nV1 — Forecast only")

    print(
        f"  Under-provisioned: "
        f"{v1_metrics['under_provisioned_percent']:.4f}%"
    )

    print(
        f"  Mean shortage: "
        f"{v1_metrics['mean_shortage']:.2f}"
    )

    print(
        f"  P95 shortage: "
        f"{v1_metrics['p95_shortage']:.2f}"
    )

    print(
        f"  CVaR95 shortage: "
        f"{v1_metrics['shortfall_cvar95']:.2f}"
    )

    print(
        f"  Mean unused capacity: "
        f"{v1_metrics['mean_unused_capacity']:.2f}"
    )

    print(
        f"  Mean provisioned capacity: "
        f"{v1_metrics['mean_provisioned_capacity']:.2f}"
    )

    print(
        f"  Reserve activation: "
        f"{v1_metrics['reserve_activation_rate']:.4f}%"
    )

    print("\nV2 — Fixed risk")

    print(
        f"  Under-provisioned: "
        f"{v2_metrics['under_provisioned_percent']:.4f}%"
    )

    print(
        f"  Mean shortage: "
        f"{v2_metrics['mean_shortage']:.2f}"
    )

    print(
        f"  P95 shortage: "
        f"{v2_metrics['p95_shortage']:.2f}"
    )

    print(
        f"  CVaR95 shortage: "
        f"{v2_metrics['shortfall_cvar95']:.2f}"
    )

    print(
        f"  Mean unused capacity: "
        f"{v2_metrics['mean_unused_capacity']:.2f}"
    )

    print(
        f"  Mean provisioned capacity: "
        f"{v2_metrics['mean_provisioned_capacity']:.2f}"
    )

    print(
        f"  Reserve activation: "
        f"{v2_metrics['reserve_activation_rate']:.4f}%"
    )

    print("\nV3 — Adaptive risk")

    print(
        f"  Under-provisioned: "
        f"{v3_metrics['under_provisioned_percent']:.4f}%"
    )

    print(
        f"  Mean shortage: "
        f"{v3_metrics['mean_shortage']:.2f}"
    )

    print(
        f"  P95 shortage: "
        f"{v3_metrics['p95_shortage']:.2f}"
    )

    print(
        f"  CVaR95 shortage: "
        f"{v3_metrics['shortfall_cvar95']:.2f}"
    )

    print(
        f"  Mean unused capacity: "
        f"{v3_metrics['mean_unused_capacity']:.2f}"
    )

    print(
        f"  Mean provisioned capacity: "
        f"{v3_metrics['mean_provisioned_capacity']:.2f}"
    )

    print(
        f"  Reserve activation: "
        f"{v3_metrics['reserve_activation_rate']:.4f}%"
    )

    # ------------------------------------------------------------------
    # 16. SAVE POLICY TABLES
    # ------------------------------------------------------------------

    print(
        "\n[16] Saving policy results..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    common_columns = [
        "interval",
        "split",
        ACTUAL_COLUMN,
        HGB_COLUMN,
        CP_HGB_COLUMN,
        "actual_workload",
        "forecast_workload",
        "risk_signal",
        "adaptive_threshold"
    ]

    simulation[
        common_columns
        + [
            "V1_reserve_trigger",
            "V1_provisioned_capacity"
        ]
    ].to_csv(
        V1_OUTPUT,
        index=False
    )

    simulation[
        common_columns
        + [
            "V2_reserve_trigger",
            "V2_provisioned_capacity"
        ]
    ].to_csv(
        V2_OUTPUT,
        index=False
    )

    simulation[
        common_columns
        + [
            "V3_reserve_trigger",
            "V3_provisioned_capacity"
        ]
    ].to_csv(
        V3_OUTPUT,
        index=False
    )

    print(
        f"V1 saved:\n  {V1_OUTPUT}"
    )

    print(
        f"V2 saved:\n  {V2_OUTPUT}"
    )

    print(
        f"V3 saved:\n  {V3_OUTPUT}"
    )

    # ------------------------------------------------------------------
    # 17. SAVE METADATA
    # ------------------------------------------------------------------

    print(
        "\n[17] Saving simulation metadata..."
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = {

        "module":
            "10_capacity_simulation",

        "capacity_definition":
            "normalized workload capacity",

        "capacity_unit":
            "observed workload-token units",

        "token_to_gpu_conversion":
            False,

        "primary_forecast":
            "CP-HGB",

        "workload_source":
            str(
                WORKLOAD_PATH.relative_to(
                    BASE_DIR
                )
            ),

        "risk_source":
            str(
                RISK_PATH.relative_to(
                    BASE_DIR
                )
            ),

        "adaptive_threshold_source":
            str(
                ADAPTIVE_THRESHOLD_PATH.relative_to(
                    BASE_DIR
                )
            ),

        "risk_aggregation":
            "maximum calibrated minute-level risk "
            "within each five-minute interval",

        "adaptive_threshold_aggregation":
            "first threshold available within "
            "each five-minute interval",

        "fixed_threshold":
            FIXED_THRESHOLD,

        "reserve_factor":
            RESERVE_FACTOR,

        "simulation_intervals":
            int(len(simulation)),

        "V1_metrics":
            v1_metrics,

        "V2_metrics":
            v2_metrics,

        "V3_metrics":
            v3_metrics,

        "change_point_reset":
            False,

        "test_parameter_tuning":
            False,

        "production_autoscaling_claim":
            False
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
        f"Metadata saved:\n  {METADATA_OUTPUT}"
    )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CAPACITY SIMULATION COMPLETE")
    print("=" * 70)

    print("\nCreated:")

    print(
        "  results/policies/V1_forecast_only.csv"
    )

    print(
        "  results/policies/V2_fixed_risk.csv"
    )

    print(
        "  results/policies/V3_adaptive_risk.csv"
    )

    print(
        "  results/validation/"
        "capacity_simulation_metadata.json"
    )

    print("\nImportant:")

    print(
        "  Capacity is normalized workload capacity."
    )

    print(
        "  No token-to-GPU conversion is claimed."
    )

    print(
        "  V4 change-point reset is NOT implemented."
    )

    print(
        "  Test data was NOT used for parameter selection."
    )

    print(
        "\n10 COMPLETE"
    )


if __name__ == "__main__":
    main()