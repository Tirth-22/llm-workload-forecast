import json
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# PATHS
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

WORKLOAD_PREDICTIONS_PATH = (
    BASE_DIR
    / "results"
    / "predictions"
    / "workload_forecast_predictions.csv"
)

V1_PATH = (
    BASE_DIR
    / "results"
    / "policies"
    / "V1_forecast_only.csv"
)

V2_PATH = (
    BASE_DIR
    / "results"
    / "policies"
    / "V2_fixed_risk.csv"
)

V3_PATH = (
    BASE_DIR
    / "results"
    / "policies"
    / "V3_adaptive_risk.csv"
)

V4_PATH = (
    BASE_DIR
    / "results"
    / "policies"
    / "V4_adaptive_reset.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "tables"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "bootstrap_tests.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "bootstrap_tests.json"
)


# ======================================================================
# BOOTSTRAP CONFIGURATION
# ======================================================================

N_BOOTSTRAP = 2000

# Paper 3 methodology:
# 288 five-minute observations = one day.
BLOCK_LENGTH = 288

CONFIDENCE_LEVEL = 0.95

ALPHA = 1.0 - CONFIDENCE_LEVEL

RANDOM_SEED = 42


# ======================================================================
# HELPERS
# ======================================================================

def percentile_interval(
    bootstrap_values,
    confidence=0.95
):

    lower = (
        (1.0 - confidence)
        / 2.0
        * 100.0
    )

    upper = (
        (1.0 + confidence)
        / 2.0
        * 100.0
    )

    return (
        float(
            np.percentile(
                bootstrap_values,
                lower
            )
        ),
        float(
            np.percentile(
                bootstrap_values,
                upper
            )
        )
    )


def moving_block_bootstrap_mean_difference(
    differences,
    n_bootstrap,
    block_length,
    rng
):
    """
    Moving-block bootstrap for a mean paired difference.

    The original observations are kept in chronological order.
    Each bootstrap sample is constructed by sampling contiguous
    blocks with replacement.

    Returns bootstrap estimates of the mean difference.
    """

    differences = np.asarray(
        differences,
        dtype=float
    )

    n = len(differences)

    if n == 0:
        raise ValueError(
            "Cannot bootstrap an empty difference array."
        )

    if block_length > n:
        raise ValueError(
            f"Block length {block_length} exceeds "
            f"sample size {n}."
        )

    # Possible block starting positions.
    starts = np.arange(
        0,
        n - block_length + 1
    )

    # Number of blocks required to cover the sample.
    n_blocks = int(
        np.ceil(
            n / block_length
        )
    )

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float
    )

    for b in range(n_bootstrap):

        selected_starts = rng.choice(
            starts,
            size=n_blocks,
            replace=True
        )

        pieces = []

        for start in selected_starts:

            pieces.append(
                differences[
                    start:
                    start + block_length
                ]
            )

        sample = np.concatenate(
            pieces
        )[:n]

        bootstrap_means[b] = (
            np.mean(sample)
        )

    return bootstrap_means


def bootstrap_comparison(
    differences,
    comparison,
    metric,
    rng
):
    """
    Perform a paired moving-block bootstrap.

    Difference definition:

        policy/model A - policy/model B

    Therefore:

        negative value
        -> A has a lower metric than B

        positive value
        -> A has a higher metric than B
    """

    differences = np.asarray(
        differences,
        dtype=float
    )

    observed = float(
        np.mean(differences)
    )

    bootstrap_values = (
        moving_block_bootstrap_mean_difference(
            differences,
            N_BOOTSTRAP,
            BLOCK_LENGTH,
            rng
        )
    )

    ci_low, ci_high = (
        percentile_interval(
            bootstrap_values,
            CONFIDENCE_LEVEL
        )
    )

    # Two-sided bootstrap sign probability.
    #
    # This is a simple empirical two-sided p-value around zero.
    # It is used only for uncertainty assessment and then corrected
    # using Holm's procedure across the planned comparisons.

    proportion_nonnegative = (
        np.mean(
            bootstrap_values >= 0
        )
    )

    proportion_nonpositive = (
        np.mean(
            bootstrap_values <= 0
        )
    )

    p_value = float(
        2.0
        *
        min(
            proportion_nonnegative,
            proportion_nonpositive
        )
    )

    p_value = min(
        p_value,
        1.0
    )

    return {
        "comparison":
            comparison,

        "metric":
            metric,

        "n_observations":
            int(len(differences)),

        "block_length":
            BLOCK_LENGTH,

        "observed_difference":
            observed,

        "ci_95_lower":
            ci_low,

        "ci_95_upper":
            ci_high,

        "raw_p_value":
            p_value,

        "bootstrap_replicates":
            N_BOOTSTRAP
    }


# ======================================================================
# HOLM CORRECTION
# ======================================================================

def holm_correction(
    p_values,
    alpha=0.05
):
    """
    Holm-Bonferroni step-down correction.

    Returns adjusted p-values and rejection decisions.
    """

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    m = len(p_values)

    order = np.argsort(
        p_values
    )

    adjusted = np.empty(
        m,
        dtype=float
    )

    running_max = 0.0

    for rank, index in enumerate(order):

        adjusted_value = (
            (m - rank)
            *
            p_values[index]
        )

        running_max = max(
            running_max,
            adjusted_value
        )

        adjusted[index] = min(
            running_max,
            1.0
        )

    reject = (
        adjusted
        <
        alpha
    )

    return adjusted, reject


# ======================================================================
# CAPACITY DERIVED ARRAYS
# ======================================================================

def load_policy(
    path,
    policy_name
):

    if not path.exists():

        raise FileNotFoundError(
            f"{policy_name} file not found:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    required = [
        "interval",
        "split",
        "actual_workload"
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{policy_name} missing columns:\n"
            + "\n".join(missing)
        )

    df = df[
        df["split"]
        .astype(str)
        .str.lower()
        == "test"
    ].copy()

    if len(df) == 0:

        raise ValueError(
            f"{policy_name} has no test rows."
        )

    df["interval"] = (
        df["interval"]
        .astype(int)
    )

    df = (
        df
        .sort_values("interval")
        .reset_index(drop=True)
    )

    return df


def derive_capacity_arrays(
    df,
    policy_name
):

    if policy_name == "V1":

        provisioned_column = (
            "V1_provisioned_capacity"
        )

        reserve_column = (
            "V1_reserve_trigger"
        )

    elif policy_name == "V2":

        provisioned_column = (
            "V2_provisioned_capacity"
        )

        reserve_column = (
            "V2_reserve_trigger"
        )

    elif policy_name == "V3":

        provisioned_column = (
            "V3_provisioned_capacity"
        )

        reserve_column = (
            "V3_reserve_trigger"
        )

    elif policy_name == "V4":

        provisioned_column = (
            "V4_provisioned_capacity"
        )

        reserve_column = (
            "V4_reserve_trigger"
        )

    else:

        raise ValueError(
            f"Unknown policy: {policy_name}"
        )

    required = [
        provisioned_column,
        reserve_column
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{policy_name} missing columns:\n"
            + "\n".join(missing)
        )

    actual = (
        df["actual_workload"]
        .astype(float)
        .to_numpy()
    )

    provisioned = (
        df[provisioned_column]
        .astype(float)
        .to_numpy()
    )

    reserve = (
        df[reserve_column]
        .astype(float)
        .to_numpy()
    )

    shortage = np.maximum(
        actual - provisioned,
        0.0
    )

    unused = np.maximum(
        provisioned - actual,
        0.0
    )

    return {
        "interval":
            df["interval"].to_numpy(),

        "actual":
            actual,

        "provisioned":
            provisioned,

        "shortage":
            shortage,

        "unused":
            unused,

        "reserve":
            reserve
    }


# ======================================================================
# ALIGN POLICY ARRAYS
# ======================================================================

def align_two_policies(
    df_a,
    df_b,
    name_a,
    name_b
):

    merged = pd.merge(
        df_a,
        df_b,
        on="interval",
        how="inner",
        suffixes=(
            f"_{name_a}",
            f"_{name_b}"
        ),
        validate="one_to_one"
    )

    if len(merged) == 0:

        raise ValueError(
            f"No overlapping test intervals between "
            f"{name_a} and {name_b}."
        )

    return merged


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("MODULE 13 — MOVING-BLOCK BOOTSTRAP TESTS")
    print("=" * 70)

    print(
        f"\nBootstrap replicates: {N_BOOTSTRAP:,}"
    )

    print(
        f"Block length: {BLOCK_LENGTH}"
    )

    print(
        "Confidence level: 95%"
    )

    print(
        f"Random seed: {RANDOM_SEED}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    results = []

    # ==================================================================
    # PART A — FORECAST COMPARISON
    # ==================================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PART A — HGB VS CP-HGB"
    )

    print(
        "=" * 70
    )

    if not WORKLOAD_PREDICTIONS_PATH.exists():

        raise FileNotFoundError(
            "Workload prediction file not found."
        )

    forecast = pd.read_csv(
        WORKLOAD_PREDICTIONS_PATH
    )

    forecast = forecast[
        forecast["split"]
        .astype(str)
        .str.lower()
        == "test"
    ].copy()

    forecast = (
        forecast
        .sort_values("interval")
        .reset_index(drop=True)
    )

    required_forecast = [
        "interval",
        "actual_next_5min_tokens",
        "hgb_prediction",
        "cp_hgb_prediction"
    ]

    missing = [
        c
        for c in required_forecast
        if c not in forecast.columns
    ]

    if missing:

        raise ValueError(
            "Missing forecast columns:\n"
            + "\n".join(missing)
        )

    actual = (
        forecast[
            "actual_next_5min_tokens"
        ]
        .astype(float)
        .to_numpy()
    )

    hgb = (
        forecast[
            "hgb_prediction"
        ]
        .astype(float)
        .to_numpy()
    )

    cp_hgb = (
        forecast[
            "cp_hgb_prediction"
        ]
        .astype(float)
        .to_numpy()
    )

    hgb_abs_error = np.abs(
        actual - hgb
    )

    cp_hgb_abs_error = np.abs(
        actual - cp_hgb
    )

    mae_difference = (
        hgb_abs_error
        -
        cp_hgb_abs_error
    )

    result = bootstrap_comparison(
        mae_difference,
        "HGB - CP-HGB",
        "MAE",
        rng
    )

    results.append(result)

    print(
        f"\nForecast MAE difference "
        f"(HGB - CP-HGB): "
        f"{result['observed_difference']:.6f}"
    )

    print(
        f"95% CI: "
        f"[{result['ci_95_lower']:.6f}, "
        f"{result['ci_95_upper']:.6f}]"
    )

    print(
        f"Raw p-value: "
        f"{result['raw_p_value']:.6f}"
    )

    # ------------------------------------------------------------------
    # RMSE comparison
    #
    # We compare per-observation squared errors and bootstrap their
    # difference. This is a paired error comparison rather than
    # pretending RMSE observations themselves are independent.
    # ------------------------------------------------------------------

    hgb_squared_error = (
        actual - hgb
    ) ** 2

    cp_hgb_squared_error = (
        actual - cp_hgb
    ) ** 2

    rmse_difference = (
        hgb_squared_error
        -
        cp_hgb_squared_error
    )

    result = bootstrap_comparison(
        rmse_difference,
        "HGB - CP-HGB",
        "Squared_Error",
        rng
    )

    results.append(result)

    print(
        f"\nForecast squared-error difference "
        f"(HGB - CP-HGB): "
        f"{result['observed_difference']:.6f}"
    )

    print(
        f"95% CI: "
        f"[{result['ci_95_lower']:.6f}, "
        f"{result['ci_95_upper']:.6f}]"
    )

    print(
        f"Raw p-value: "
        f"{result['raw_p_value']:.6f}"
    )

    # ==================================================================
    # LOAD POLICIES
    # ==================================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOADING CAPACITY POLICIES"
    )

    print(
        "=" * 70
    )

    policy_paths = {
        "V1": V1_PATH,
        "V2": V2_PATH,
        "V3": V3_PATH,
        "V4": V4_PATH
    }

    policies = {}

    for name, path in policy_paths.items():

        policies[name] = load_policy(
            path,
            name
        )

        print(
            f"{name}: "
            f"{len(policies[name]):,} test intervals"
        )

    # ==================================================================
    # CAPACITY COMPARISON FUNCTION
    # ==================================================================

    def compare_capacity(
        name_a,
        name_b
    ):

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"{name_a} VS {name_b}"
        )

        print(
            "-" * 70
        )

        merged = align_two_policies(
            policies[name_a],
            policies[name_b],
            name_a,
            name_b
        )

        # --------------------------------------------------------------
        # Build arrays directly from merged data.
        # --------------------------------------------------------------

        actual_a = (
            merged[
                f"actual_workload_{name_a}"
            ]
            .astype(float)
            .to_numpy()
        )

        actual_b = (
            merged[
                f"actual_workload_{name_b}"
            ]
            .astype(float)
            .to_numpy()
        )

        if not np.allclose(
            actual_a,
            actual_b
        ):

            raise ValueError(
                f"Actual workload differs between "
                f"{name_a} and {name_b}."
            )

        actual = actual_a

        provisioned_a = (
            merged[
                f"{name_a}_provisioned_capacity"
            ]
            .astype(float)
            .to_numpy()
        )

        provisioned_b = (
            merged[
                f"{name_b}_provisioned_capacity"
            ]
            .astype(float)
            .to_numpy()
        )

        reserve_a = (
            merged[
                f"{name_a}_reserve_trigger"
            ]
            .astype(float)
            .to_numpy()
        )

        reserve_b = (
            merged[
                f"{name_b}_reserve_trigger"
            ]
            .astype(float)
            .to_numpy()
        )

        shortage_a = np.maximum(
            actual - provisioned_a,
            0.0
        )

        shortage_b = np.maximum(
            actual - provisioned_b,
            0.0
        )

        unused_a = np.maximum(
            provisioned_a - actual,
            0.0
        )

        unused_b = np.maximum(
            provisioned_b - actual,
            0.0
        )

        # --------------------------------------------------------------
        # Paired comparisons
        #
        # Difference = A - B
        # --------------------------------------------------------------

        comparisons = [

            (
                "Mean_Shortage",
                shortage_a - shortage_b
            ),

            (
                "Mean_Unused_Capacity",
                unused_a - unused_b
            ),

            (
                "Mean_Provisioned_Capacity",
                provisioned_a - provisioned_b
            ),

            (
                "Reserve_Activation",
                reserve_a - reserve_b
            )
        ]

        for metric, differences in comparisons:

            result = bootstrap_comparison(
                differences,
                f"{name_a} - {name_b}",
                metric,
                rng
            )

            result[
                "test_intervals"
            ] = int(len(merged))

            results.append(
                result
            )

            print(
                f"\n{metric}"
            )

            print(
                f"  Observed difference: "
                f"{result['observed_difference']:.6f}"
            )

            print(
                f"  95% CI: "
                f"[{result['ci_95_lower']:.6f}, "
                f"{result['ci_95_upper']:.6f}]"
            )

            print(
                f"  Raw p-value: "
                f"{result['raw_p_value']:.6f}"
            )

    # ==================================================================
    # REQUIRED CAPACITY COMPARISONS
    # ==================================================================

    compare_capacity(
        "V4",
        "V1"
    )

    compare_capacity(
        "V4",
        "V2"
    )

    compare_capacity(
        "V4",
        "V3"
    )

    # ==================================================================
    # HOLM CORRECTION
    # ==================================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HOLM MULTIPLE-COMPARISON CORRECTION"
    )

    print(
        "=" * 70
    )

    p_values = np.array(
        [
            row["raw_p_value"]
            for row in results
        ],
        dtype=float
    )

    adjusted_p, reject = (
        holm_correction(
            p_values,
            alpha=0.05
        )
    )

    for i, row in enumerate(results):

        row[
            "holm_adjusted_p_value"
        ] = float(
            adjusted_p[i]
        )

        row[
            "significant_after_holm_alpha_0_05"
        ] = bool(
            reject[i]
        )

    for row in results:

        print(
            f"\n{row['comparison']} | "
            f"{row['metric']}"
        )

        print(
            f"  Raw p: "
            f"{row['raw_p_value']:.6f}"
        )

        print(
            f"  Holm adjusted p: "
            f"{row['holm_adjusted_p_value']:.6f}"
        )

        print(
            f"  Significant after Holm: "
            f"{row['significant_after_holm_alpha_0_05']}"
        )

    # ==================================================================
    # SAVE RESULTS
    # ==================================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SAVING BOOTSTRAP RESULTS"
    )

    print(
        "=" * 70
    )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    metadata = {

        "module":
            "13_bootstrap_tests",

        "bootstrap_replicates":
            N_BOOTSTRAP,

        "block_length":
            BLOCK_LENGTH,

        "confidence_level":
            CONFIDENCE_LEVEL,

        "alpha":
            ALPHA,

        "random_seed":
            RANDOM_SEED,

        "method":
            "paired moving-block bootstrap",

        "block_interpretation":
            "288 five-minute observations = one day",

        "multiple_comparison_correction":
            "Holm-Bonferroni",

        "comparisons":
            [
                "HGB vs CP-HGB",
                "V4 vs V1",
                "V4 vs V2",
                "V4 vs V3"
            ],

        "test_data_used_for_parameter_tuning":
            False,

        "winner_selection":
            False
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "metadata": metadata,
                "results": results
            },
            file,
            indent=2
        )

    print(
        "\nCreated:"
    )

    print(
        "  results/tables/"
        "bootstrap_tests.csv"
    )

    print(
        "  results/tables/"
        "bootstrap_tests.json"
    )

    # ==================================================================
    # FINAL
    # ==================================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODULE 13 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nNo model was retrained."
    )

    print(
        "No test parameter was tuned."
    )

    print(
        "Paired temporal bootstrap was used."
    )

    print(
        "Holm correction was applied."
    )

    print(
        "No overall winner was declared."
    )


if __name__ == "__main__":
    main()