from pathlib import Path
import json
import pandas as pd


# ======================================================================
# PATHS
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TABLES_DIR = (
    BASE_DIR
    / "results"
    / "tables"
)

OUTPUT_DIR = TABLES_DIR


# Existing Module 12 outputs
FORECAST_PATH = (
    TABLES_DIR
    / "forecast_metrics.csv"
)

RISK_PATH = (
    TABLES_DIR
    / "risk_metrics.csv"
)

CAPACITY_PATH = (
    TABLES_DIR
    / "capacity_metrics.csv"
)

BOOTSTRAP_PATH = (
    TABLES_DIR
    / "bootstrap_tests.csv"
)

# Existing validation metadata
CHANGE_POINT_METADATA = (
    BASE_DIR
    / "results"
    / "validation"
    / "change_point_metadata.json"
)

WORKLOAD_METADATA = (
    BASE_DIR
    / "results"
    / "validation"
    / "workload_forecaster_metadata.json"
)

RISK_METADATA = (
    BASE_DIR
    / "results"
    / "validation"
    / "failure_risk_training_metadata.json"
)

CALIBRATION_METADATA = (
    BASE_DIR
    / "results"
    / "validation"
    / "calibration_metadata.json"
)

ADAPTIVE_METADATA = (
    BASE_DIR
    / "results"
    / "validation"
    / "adaptive_threshold_metadata.json"
)

CAPACITY_METADATA = (
    BASE_DIR
    / "results"
    / "validation"
    / "capacity_simulation_metadata.json"
)

RESET_METADATA = (
    BASE_DIR
    / "results"
    / "validation"
    / "change_point_reset_metadata.json"
)


# ======================================================================
# HELPERS
# ======================================================================

def load_json(path):

    if not path.exists():
        return {}

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def require_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def save_csv(
    df,
    filename
):

    path = (
        OUTPUT_DIR
        / filename
    )

    df.to_csv(
        path,
        index=False
    )

    print(
        f"Created: results/tables/{filename}"
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("MODULE 15 — RESEARCH TABLE GENERATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==================================================================
    # VALIDATE INPUTS
    # ==================================================================

    print("\nValidating Module 12/13 outputs...")

    required_files = [
        FORECAST_PATH,
        RISK_PATH,
        CAPACITY_PATH,
        BOOTSTRAP_PATH
    ]

    for path in required_files:
        require_file(path)

    print(
        "Evaluation outputs: PASS"
    )

    # ==================================================================
    # LOAD RESULTS
    # ==================================================================

    forecast = pd.read_csv(
        FORECAST_PATH
    )

    risk = pd.read_csv(
        RISK_PATH
    )

    capacity = pd.read_csv(
        CAPACITY_PATH
    )

    bootstrap = pd.read_csv(
        BOOTSTRAP_PATH
    )

    # ==================================================================
    # TABLE 1 — FORECAST PERFORMANCE
    # ==================================================================

    print(
        "\n[1] Forecast performance table..."
    )

    table1 = forecast.copy()

    table1 = table1[
        [
            "model",
            "test_rows",
            "MAE",
            "RMSE",
            "WAPE_percent",
            "sMAPE_percent",
            "mean_bias"
        ]
    ]

    save_csv(
        table1,
        "table1_forecast_performance.csv"
    )

    # ==================================================================
    # TABLE 2 — FAILURE-RISK PERFORMANCE
    # ==================================================================

    print(
        "\n[2] Failure-risk performance table..."
    )

    table2 = risk.copy()

    table2 = table2[
        [
            "test_rows",
            "failure_prevalence_percent",
            "ROC_AUC",
            "Average_Precision",
            "Brier_calibrated",
            "Brier_raw_diagnostic",
            "ECE_10_bin",
            "classification_threshold",
            "Precision",
            "Recall",
            "F1"
        ]
    ]

    save_csv(
        table2,
        "table2_failure_risk_performance.csv"
    )

    # ==================================================================
    # TABLE 3 — CAPACITY POLICY PERFORMANCE
    # ==================================================================

    print(
        "\n[3] Capacity policy table..."
    )

    table3 = capacity.copy()

    table3 = table3[
        [
            "policy",
            "intervals",
            "under_provisioned_percent",
            "mean_shortage",
            "p95_shortage",
            "shortfall_cvar95",
            "mean_unused_capacity",
            "mean_provisioned_capacity",
            "mean_actual_workload",
            "reserve_activation_percent",
            "cost_index"
        ]
    ]

    save_csv(
        table3,
        "table3_capacity_policy_performance.csv"
    )

    # ==================================================================
    # TABLE 4 — STATISTICAL TESTS
    # ==================================================================

    print(
        "\n[4] Statistical comparison table..."
    )

    table4 = bootstrap.copy()

    table4 = table4[
        [
            "comparison",
            "metric",
            "n_observations",
            "block_length",
            "observed_difference",
            "ci_95_lower",
            "ci_95_upper",
            "raw_p_value",
            "holm_adjusted_p_value",
            "significant_after_holm_alpha_0_05"
        ]
    ]

    save_csv(
        table4,
        "table4_bootstrap_statistical_tests.csv"
    )

    # ==================================================================
    # TABLE 5 — EXPERIMENT CONFIGURATION
    # ==================================================================

    print(
        "\n[5] Experiment configuration table..."
    )

    cp_meta = load_json(
        CHANGE_POINT_METADATA
    )

    workload_meta = load_json(
        WORKLOAD_METADATA
    )

    risk_meta = load_json(
        RISK_METADATA
    )

    calibration_meta = load_json(
        CALIBRATION_METADATA
    )

    adaptive_meta = load_json(
        ADAPTIVE_METADATA
    )

    capacity_meta = load_json(
        CAPACITY_METADATA
    )

    reset_meta = load_json(
        RESET_METADATA
    )

    configuration_rows = [

        {
            "component":
                "Change-point detector",

            "parameter":
                "training_threshold_percentile",

            "value":
                cp_meta.get(
                    "threshold_percentile",
                    95
                ),

            "source":
                "Paper 3 reproduction"
        },

        {
            "component":
                "Change-point detector",

            "parameter":
                "derived_threshold",

            "value":
                cp_meta.get(
                    "derived_threshold",
                    None
                ),

            "source":
                "Local training data"
        },

        {
            "component":
                "Workload forecasting",

            "parameter":
                "model",

            "value":
                "Histogram Gradient Boosting",

            "source":
                "Paper 3 reproduction"
        },

        {
            "component":
                "Failure-risk model",

            "parameter":
                "model",

            "value":
                "XGBoost",

            "source":
                "Paper 4 reproduction"
        },

        {
            "component":
                "Calibration",

            "parameter":
                "method",

            "value":
                "Isotonic calibration",

            "source":
                "Paper 4 reproduction"
        },

        {
            "component":
                "Adaptive threshold",

            "parameter":
                "base_threshold",

            "value":
                adaptive_meta.get(
                    "theta_base",
                    0.9591
                ),

            "source":
                "Paper 4 validation-selected threshold"
        },

        {
            "component":
                "Adaptive threshold",

            "parameter":
                "eta",

            "value":
                adaptive_meta.get(
                    "eta",
                    0.01
                ),

            "source":
                "Experiment configuration"
        },

        {
            "component":
                "Adaptive threshold",

            "parameter":
                "minimum_threshold",

            "value":
                adaptive_meta.get(
                    "theta_min",
                    0.50
                ),

            "source":
                "Experiment configuration"
        },

        {
            "component":
                "Adaptive threshold",

            "parameter":
                "maximum_threshold",

            "value":
                adaptive_meta.get(
                    "theta_max",
                    0.99
                ),

            "source":
                "Experiment configuration"
        },

        {
            "component":
                "Capacity reserve",

            "parameter":
                "reserve_factor",

            "value":
                capacity_meta.get(
                    "reserve_factor",
                    reset_meta.get(
                        "reserve_factor",
                        1.10
                    )
                ),

            "source":
                "Experiment configuration"
        },

        {
            "component":
                "Bootstrap",

            "parameter":
                "replicates",

            "value":
                2000,

            "source":
                "Paper 3 methodology"
        },

        {
            "component":
                "Bootstrap",

            "parameter":
                "block_length",

            "value":
                288,

            "source":
                "Paper 3 methodology"
        },

        {
            "component":
                "Bootstrap",

            "parameter":
                "confidence_level",

            "value":
                0.95,

            "source":
                "Paper 3 methodology"
        },

        {
            "component":
                "Multiple comparisons",

            "parameter":
                "correction",

            "value":
                "Holm-Bonferroni",

            "source":
                "Experiment methodology"
        }
    ]

    table5 = pd.DataFrame(
        configuration_rows
    )

    save_csv(
        table5,
        "table5_experiment_configuration.csv"
    )

    # ==================================================================
    # TABLE 6 — REPRODUCTION VS PROPOSED COMPONENTS
    # ==================================================================

    print(
        "\n[6] Reproduction/proposed-work table..."
    )

    table6 = pd.DataFrame(
        [

            {
                "component":
                    "BurstGPT preprocessing",

                "category":
                    "Reproduced",

                "basis":
                    "BurstGPT / Paper 3 / Paper 4"
            },

            {
                "component":
                    "Five-minute workload aggregation",

                "category":
                    "Reproduced",

                "basis":
                    "Paper 3"
            },

            {
                "component":
                    "Causal change-point detector",

                "category":
                    "Reproduced",

                "basis":
                    "Paper 3"
            },

            {
                "component":
                    "HGB workload forecaster",

                "category":
                    "Reproduced",

                "basis":
                    "Paper 3"
            },

            {
                "component":
                    "CP-HGB workload forecaster",

                "category":
                    "Reproduced",

                "basis":
                    "Paper 3"
            },

            {
                "component":
                    "Minute-level failure target",

                "category":
                    "Reproduced",

                "basis":
                    "Paper 4"
            },

            {
                "component":
                    "89 risk features",

                "category":
                    "Reproduced",

                "basis":
                    "Paper 4"
            },

            {
                "component":
                    "XGBoost failure-risk model",

                "category":
                    "Reproduced",

                "basis":
                    "Paper 4"
            },

            {
                "component":
                    "Isotonic probability calibration",

                "category":
                    "Reproduced",

                "basis":
                    "Paper 4"
            },

            {
                "component":
                    "Forecast-driven normalized capacity",

                "category":
                    "Integration",

                "basis":
                    "Project methodology"
            },

            {
                "component":
                    "Adaptive risk threshold",

                "category":
                    "Proposed",

                "basis":
                    "Project methodology"
            },

            {
                "component":
                    "Change-point threshold reset",

                "category":
                    "Proposed",

                "basis":
                    "Project methodology"
            },

            {
                "component":
                    "Four-policy comparison",

                "category":
                    "Proposed",

                "basis":
                    "Project methodology"
            }
        ]
    )

    save_csv(
        table6,
        "table6_reproduction_vs_proposed.csv"
    )

    # ==================================================================
    # TABLE 7 — POLICY DEFINITIONS
    # ==================================================================

    print(
        "\n[7] Policy definition table..."
    )

    table7 = pd.DataFrame(
        [

            {
                "policy":
                    "V1",

                "description":
                    "CP-HGB forecast-driven capacity with no risk reserve",

                "risk_threshold":
                    "None",

                "reserve":
                    "None",

                "change_point_reset":
                    "No"
            },

            {
                "policy":
                    "V2",

                "description":
                    "CP-HGB forecast with fixed failure-risk threshold",

                "risk_threshold":
                    "Fixed Paper 4 base threshold",

                "reserve":
                    "10%",

                "change_point_reset":
                    "No"
            },

            {
                "policy":
                    "V3",

                "description":
                    "CP-HGB forecast with adaptive failure-risk threshold",

                "risk_threshold":
                    "Adaptive",

                "reserve":
                    "10%",

                "change_point_reset":
                    "No"
            },

            {
                "policy":
                    "V4",

                "description":
                    "CP-HGB forecast with adaptive threshold and change-point reset",

                "risk_threshold":
                    "Adaptive + reset",

                "reserve":
                    "10%",

                "change_point_reset":
                    "Yes"
            }
        ]
    )

    save_csv(
        table7,
        "table7_policy_definitions.csv"
    )

    # ==================================================================
    # TABLE 8 — RESEARCH HYPOTHESIS STATUS
    # ==================================================================

    print(
        "\n[8] Hypothesis evidence table..."
    )

    table8 = pd.DataFrame(
        [

            {
                "hypothesis":
                    "H1",

                "statement":
                    "CP-HGB will exhibit different forecasting performance from HGB under workload regime changes.",

                "current_evidence":
                    "Observed differences exist; bootstrap results should be interpreted with Holm-adjusted uncertainty.",

                "status":
                    "Not declared as accepted/rejected by this table"
            },

            {
                "hypothesis":
                    "H2",

                "statement":
                    "Adaptive risk thresholds will produce a different reliability-capacity trade-off from a fixed threshold.",

                "current_evidence":
                    "V2 and V3 show different reserve activation and capacity values.",

                "status":
                    "Requires interpretation with statistical results"
            },

            {
                "hypothesis":
                    "H3",

                "statement":
                    "Adding the change-point reset to the adaptive threshold will change performance during detected regime shifts.",

                "current_evidence":
                    "V3 and V4 are identical on current aggregate locked-test capacity metrics.",

                "status":
                    "No aggregate improvement demonstrated"
            },

            {
                "hypothesis":
                    "H4",

                "statement":
                    "V4 will reduce capacity shortfall relative to a no-reserve baseline without requiring the same reserve level continuously.",

                "current_evidence":
                    "V4 has nonzero reserve activation while V1 has zero; aggregate shortage difference requires statistical interpretation.",

                "status":
                    "Requires statistical interpretation"
            }
        ]
    )

    save_csv(
        table8,
        "table8_hypothesis_evidence.csv"
    )

    # ==================================================================
    # FINAL SUMMARY JSON
    # ==================================================================

    print(
        "\nGenerating table manifest..."
    )

    manifest = {

        "module":
            "15_make_tables",

        "tables_created":
            8,

        "tables": [

            "table1_forecast_performance.csv",

            "table2_failure_risk_performance.csv",

            "table3_capacity_policy_performance.csv",

            "table4_bootstrap_statistical_tests.csv",

            "table5_experiment_configuration.csv",

            "table6_reproduction_vs_proposed.csv",

            "table7_policy_definitions.csv",

            "table8_hypothesis_evidence.csv"
        ],

        "source_modules": [
            "03_change_point_detector",
            "05_train_short_term_forecaster",
            "07_train_failure_risk",
            "08_calibration",
            "09_adaptive_threshold",
            "10_capacity_simulation",
            "11_change_point_reset",
            "12_metrics",
            "13_bootstrap_tests",
            "14_make_figures"
        ],

        "test_data_used_for_training":
            False,

        "test_parameter_tuning":
            False,

        "winner_declared":
            False
    }

    manifest_path = (
        TABLES_DIR
        / "table_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2
        )

    print(
        "\nCreated:"
    )

    print(
        "  results/tables/table_manifest.json"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODULE 15 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nAll paper-oriented tables generated."
    )

    print(
        "No model was retrained."
    )

    print(
        "No test parameter was tuned."
    )

    print(
        "No overall policy winner was declared."
    )


if __name__ == "__main__":
    main()