"""
Module 22 — Paired Statistical Comparison of Forecast-Driven Policies

Research topic:
Short-Term LLM Workload Prediction for Adaptive Cloud Computing

Purpose
-------
Statistically compare the three forecast-driven policies from Module 21
on the SAME locked-test intervals:

    Persistence
    HGB
    CP-HGB

The analysis is paired because every model makes a decision for the same
locked-test interval.

Primary paired quantities:
    shortage
    unused capacity
    provisioned capacity
    absolute control error

Tests:
    - Wilcoxon signed-rank tests for paired differences
    - Paired bootstrap confidence intervals for mean differences
    - Holm correction across each metric's three pairwise comparisons

No model, threshold, or controller parameter is tuned here.

This is an inference module only. It does not alter the policy results.
"""

from pathlib import Path
import json
import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "results"
    / "policies"
    / "module21_forecast_adaptive_risk_comparison.csv"
)

TABLE_DIR = ROOT / "results" / "tables"
VALIDATION_DIR = ROOT / "results" / "validation"

PAIRWISE_OUTPUT = (
    TABLE_DIR
    / "module22_paired_policy_statistics.csv"
)

SUMMARY_OUTPUT = (
    TABLE_DIR
    / "module22_paired_policy_summary.csv"
)

METADATA_OUTPUT = (
    VALIDATION_DIR
    / "module22_metadata.json"
)

TABLE_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


MODELS = [
    "Persistence",
    "HGB",
    "CP-HGB",
]

METRICS = {
    "shortage": "shortage",
    "unused_capacity": "unused_capacity",
    "provisioned_capacity": "provisioned_capacity",
    "absolute_control_error": None,
}

N_BOOTSTRAP = 10000
RANDOM_SEED = 42


def holm_adjust(p_values):
    """
    Holm-Bonferroni adjustment.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)

    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0

    for rank, idx in enumerate(order):
        value = min(
            1.0,
            (m - rank) * p_values[idx],
        )
        running_max = max(running_max, value)
        adjusted[idx] = running_max

    return adjusted


def bootstrap_mean_ci(differences, rng):
    """
    Paired bootstrap CI for the mean difference.
    """
    differences = np.asarray(
        differences,
        dtype=float,
    )

    n = len(differences)

    if n == 0:
        return np.nan, np.nan

    indices = rng.integers(
        0,
        n,
        size=(N_BOOTSTRAP, n),
    )

    bootstrap_means = (
        differences[indices]
        .mean(axis=1)
    )

    lower, upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    return float(lower), float(upper)


def compute_metric(df, metric):
    if metric == "absolute_control_error":
        return np.abs(
            df["provisioned_capacity"]
            - df["actual_workload"]
        )

    return df[metric].to_numpy(dtype=float)


def prepare_policy_matrix(df):
    """
    Convert long Module 21 output into one row per interval and model.
    """
    required = {
        "interval",
        "forecast_model",
        "actual_workload",
        "provisioned_capacity",
        "shortage",
        "unused_capacity",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Module 21 output is missing:\n"
            + "\n".join(
                f"  - {x}"
                for x in missing
            )
        )

    counts = (
        df.groupby(
            ["interval", "forecast_model"]
        )
        .size()
    )

    if (counts != 1).any():
        raise ValueError(
            "Expected exactly one policy row per "
            "interval/model."
        )

    if set(df["forecast_model"]) != set(MODELS):
        raise ValueError(
            "Unexpected forecast models found: "
            f"{sorted(df['forecast_model'].unique())}"
        )

    return df


def main():

    print("=" * 70)
    print("MODULE 22 — PAIRED POLICY STATISTICS")
    print("=" * 70)

    print("\n[1] Loading Module 21 policy results...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Module 21 output not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(
        f"Rows loaded: {len(df):,}"
    )

    prepare_policy_matrix(df)

    print(
        "Module 21 schema validation: PASS"
    )

    # --------------------------------------------------------------
    # Exact common interval intersection
    # --------------------------------------------------------------

    print("\n[2] Checking paired locked-test intervals...")

    interval_sets = {}

    for model in MODELS:
        interval_sets[model] = set(
            df.loc[
                df["forecast_model"] == model,
                "interval",
            ].astype(int)
        )

    common_intervals = set.intersection(
        *interval_sets.values()
    )

    if not common_intervals:
        raise ValueError(
            "No common intervals across all forecast models."
        )

    if not all(
        interval_sets[model] == common_intervals
        for model in MODELS
    ):
        raise ValueError(
            "Forecast models do not share exactly the same "
            "locked-test interval set."
        )

    common_intervals = sorted(
        common_intervals
    )

    print(
        f"Common paired intervals: "
        f"{len(common_intervals):,}"
    )

    # --------------------------------------------------------------
    # Build matrices
    # --------------------------------------------------------------

    paired = (
        df[
            df["interval"].isin(
                common_intervals
            )
        ]
        .sort_values(
            ["interval", "forecast_model"]
        )
        .copy()
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    pairwise_rows = []

    pairs = list(
        itertools.combinations(
            MODELS,
            2,
        )
    )

    print("\n[3] Running paired statistical comparisons...")

    for metric_name in METRICS:

        print(
            f"\nMetric: {metric_name}"
        )

        raw_p_values = []
        temp_rows = []

        for model_a, model_b in pairs:

            a = (
                paired[
                    paired["forecast_model"] == model_a
                ]
                .set_index("interval")
            )

            b = (
                paired[
                    paired["forecast_model"] == model_b
                ]
                .set_index("interval")
            )

            if metric_name == "absolute_control_error":
                values_a = np.abs(
                    a["provisioned_capacity"]
                    - a["actual_workload"]
                ).to_numpy(dtype=float)

                values_b = np.abs(
                    b["provisioned_capacity"]
                    - b["actual_workload"]
                ).to_numpy(dtype=float)

            else:
                values_a = a[
                    METRICS[metric_name]
                ].to_numpy(dtype=float)

                values_b = b[
                    METRICS[metric_name]
                ].to_numpy(dtype=float)

            if not np.array_equal(
                a.index.to_numpy(),
                b.index.to_numpy(),
            ):
                raise ValueError(
                    f"Interval alignment failed for "
                    f"{model_a} vs {model_b}."
                )

            difference = (
                values_a - values_b
            )

            nonzero = (
                np.abs(difference) > 0
            )

            if nonzero.any():
                try:
                    statistic, p_value = wilcoxon(
                        difference,
                        zero_method="wilcox",
                        alternative="two-sided",
                        method="auto",
                    )
                except ValueError:
                    statistic = np.nan
                    p_value = 1.0
            else:
                statistic = 0.0
                p_value = 1.0

            ci_low, ci_high = (
                bootstrap_mean_ci(
                    difference,
                    rng,
                )
            )

            mean_a = float(
                np.mean(values_a)
            )

            mean_b = float(
                np.mean(values_b)
            )

            mean_difference = float(
                np.mean(difference)
            )

            median_difference = float(
                np.median(difference)
            )

            temp_rows.append(
                {
                    "metric": metric_name,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_pairs": len(difference),
                    "mean_model_a": mean_a,
                    "mean_model_b": mean_b,
                    "mean_difference_a_minus_b":
                        mean_difference,
                    "median_difference_a_minus_b":
                        median_difference,
                    "bootstrap_ci95_low":
                        ci_low,
                    "bootstrap_ci95_high":
                        ci_high,
                    "wilcoxon_statistic":
                        float(statistic)
                        if np.isfinite(statistic)
                        else np.nan,
                    "raw_p_value":
                        float(p_value),
                }
            )

            raw_p_values.append(
                float(p_value)
            )

        adjusted = holm_adjust(
            raw_p_values
        )

        for row, adjusted_p in zip(
            temp_rows,
            adjusted,
        ):
            row["holm_adjusted_p_value"] = float(
                adjusted_p
            )

            row["significant_alpha_0_05"] = bool(
                adjusted_p < 0.05
            )

            pairwise_rows.append(row)

            print(
                f"  {row['model_a']} vs {row['model_b']}: "
                f"mean Δ={row['mean_difference_a_minus_b']:.4f}, "
                f"95% CI="
                f"[{row['bootstrap_ci95_low']:.4f}, "
                f"{row['bootstrap_ci95_high']:.4f}], "
                f"Holm p={adjusted_p:.6f}"
            )

    pairwise_df = pd.DataFrame(
        pairwise_rows
    )

    # --------------------------------------------------------------
    # Compact model-level summary
    # --------------------------------------------------------------

    print("\n[4] Building model-level summary...")

    summary_rows = []

    for model in MODELS:

        model_df = paired[
            paired["forecast_model"] == model
        ]

        actual = (
            model_df["actual_workload"]
            .to_numpy(dtype=float)
        )

        provisioned = (
            model_df["provisioned_capacity"]
            .to_numpy(dtype=float)
        )

        shortage = (
            model_df["shortage"]
            .to_numpy(dtype=float)
        )

        unused = (
            model_df["unused_capacity"]
            .to_numpy(dtype=float)
        )

        absolute_error = np.abs(
            provisioned - actual
        )

        summary_rows.append(
            {
                "forecast_model": model,
                "intervals": len(model_df),
                "mean_shortage": float(
                    np.mean(shortage)
                ),
                "mean_unused_capacity": float(
                    np.mean(unused)
                ),
                "mean_provisioned_capacity": float(
                    np.mean(provisioned)
                ),
                "mean_absolute_control_error": float(
                    np.mean(absolute_error)
                ),
                "under_provisioned_percent": float(
                    np.mean(shortage > 0) * 100
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    pairwise_df.to_csv(
        PAIRWISE_OUTPUT,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    metadata = {
        "module":
            "22_paired_policy_statistics",
        "source":
            str(
                INPUT_PATH.relative_to(ROOT)
            ),
        "models":
            MODELS,
        "paired_intervals":
            len(common_intervals),
        "metrics":
            list(METRICS.keys()),
        "tests":
            [
                "Wilcoxon signed-rank",
                "paired bootstrap 95% CI for mean difference",
                "Holm correction within each metric",
            ],
        "bootstrap_replicates":
            N_BOOTSTRAP,
        "random_seed":
            RANDOM_SEED,
        "test_parameter_tuning":
            False,
        "purpose":
            "Determine whether forecast-driven policy differences are statistically distinguishable on the locked test set.",
    }

    with open(
        METADATA_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("MODULE 22 RESULT")
    print("=" * 70)

    print("\nModel summary:")
    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\nCreated:")
    print(f"  {PAIRWISE_OUTPUT}")
    print(f"  {SUMMARY_OUTPUT}")
    print(f"  {METADATA_OUTPUT}")

    print("\nNo model or controller parameter was changed.")
    print("No test parameter was tuned.")
    print("MODULE 22 COMPLETE")


if __name__ == "__main__":
    main()
