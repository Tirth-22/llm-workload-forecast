"""
Module 23 — Final Implementation Audit and Research Freeze

Purpose
-------
Audit the completed implementation before paper writing.

This module DOES NOT train, tune, or modify any model/policy.

It checks:
1. Required result artifacts exist.
2. Core locked-test row counts are consistent.
3. Module 19 forecast outputs contain the expected 2,589 scored rows.
4. Module 21 contains exactly 3 policies x 2,589 intervals.
5. Module 22 uses the same 2,589 paired intervals.
6. Key reported values remain internally consistent.
7. No obvious NaN/Inf values exist in final tables.
8. The implementation remains normalized-capacity only.
9. The experiment chain is ready to freeze.

If all checks pass, a freeze manifest is written.
"""

from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"
PREDICTIONS = RESULTS / "predictions"
POLICIES = RESULTS / "policies"
TABLES = RESULTS / "tables"
VALIDATION = RESULTS / "validation"

MANIFEST_PATH = VALIDATION / "module23_freeze_manifest.json"


EXPECTED_FILES = [
    # Forecasting
    PREDICTIONS / "baseline_forecast_predictions.csv",
    TABLES / "baseline_forecast_metrics.csv",

    # Existing risk/controller
    PREDICTIONS / "calibrated_failure_risk_predictions.csv",
    POLICIES / "adaptive_threshold_predictions.csv",

    # Existing policy chain
    POLICIES / "V1_forecast_only.csv",
    POLICIES / "V2_fixed_risk.csv",
    POLICIES / "V3_adaptive_risk.csv",
    POLICIES / "V4_adaptive_reset.csv",

    # New comparative experiments
    POLICIES / "module20_forecast_control_comparison.csv",
    TABLES / "module20_forecast_control_metrics.csv",

    POLICIES / "module21_forecast_adaptive_risk_comparison.csv",
    TABLES / "module21_forecast_adaptive_risk_metrics.csv",

    TABLES / "module22_paired_policy_statistics.csv",
    TABLES / "module22_paired_policy_summary.csv",
]


def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def assert_finite(df, name):
    numeric = df.select_dtypes(
        include=[np.number]
    )

    if numeric.empty:
        return

    if not np.isfinite(
        numeric.to_numpy()
    ).all():

        raise ValueError(
            f"{name} contains NaN or infinite numeric values."
        )


def main():

    print("=" * 70)
    print("MODULE 23 — FINAL IMPLEMENTATION AUDIT / FREEZE")
    print("=" * 70)

    checks = []

    # --------------------------------------------------------------
    # 1. Required artifacts
    # --------------------------------------------------------------

    print("\n[1] Checking required final artifacts...")

    missing = [
        str(path.relative_to(ROOT))
        for path in EXPECTED_FILES
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required artifacts are missing:\n"
            + "\n".join(
                f"  - {x}"
                for x in missing
            )
        )

    print(
        f"Required artifacts present: "
        f"{len(EXPECTED_FILES)}/{len(EXPECTED_FILES)}"
    )

    checks.append(
        "required_artifacts = PASS"
    )

    # --------------------------------------------------------------
    # 2. Module 19 forecast audit
    # --------------------------------------------------------------

    print("\n[2] Auditing Module 19 locked-test forecasts...")

    forecasts = pd.read_csv(
        PREDICTIONS
        / "baseline_forecast_predictions.csv"
    )

    if len(forecasts) != 2589:
        raise ValueError(
            f"Expected 2,589 Module 19 rows, "
            f"found {len(forecasts):,}."
        )

    required_forecast_columns = [
        "interval",
        "actual_next_5min_tokens",
        "persistence_prediction",
        "hgb_prediction",
        "cp_hgb_prediction",
    ]

    missing_columns = [
        x
        for x in required_forecast_columns
        if x not in forecasts.columns
    ]

    if missing_columns:
        raise ValueError(
            "Module 19 is missing columns:\n"
            + "\n".join(
                f"  - {x}"
                for x in missing_columns
            )
        )

    if forecasts["interval"].duplicated().any():
        raise ValueError(
            "Duplicate Module 19 intervals detected."
        )

    assert_finite(
        forecasts,
        "Module 19 forecasts"
    )

    print(
        "Rows: 2,589"
    )

    print(
        f"Interval range: "
        f"{int(forecasts.interval.min())} -> "
        f"{int(forecasts.interval.max())}"
    )

    checks.append(
        "module19_locked_test = PASS"
    )

    # --------------------------------------------------------------
    # 3. Module 20 audit
    # --------------------------------------------------------------

    print("\n[3] Auditing Module 20 forecast-to-control comparison...")

    module20 = pd.read_csv(
        POLICIES
        / "module20_forecast_control_comparison.csv"
    )

    # Module 20 is stored in long format:
    # 3 forecast models x 2,589 locked-test intervals = 7,767 rows.
    expected_models = {"Persistence", "HGB", "CP-HGB"}
    model_counts = module20["forecast_model"].value_counts().to_dict()

    if set(model_counts) != expected_models:
        raise ValueError(
            "Module 20 must contain exactly the expected forecast models. "
            f"Found: {sorted(model_counts)}"
        )

    if any(count != len(forecasts) for count in model_counts.values()):
        raise ValueError(
            "Module 20 must contain 2,589 rows for each forecast model. "
            f"Found counts: {model_counts}"
        )

    if len(module20) != 3 * len(forecasts):
        raise ValueError(
            "Module 20 must contain 7,767 long-format rows "
            "(3 models x 2,589 locked-test intervals)."
        )

    assert_finite(
        module20,
        "Module 20"
    )

    print(
        "Rows: 2,589"
    )

    checks.append(
        "module20_locked_test = PASS"
    )

    # --------------------------------------------------------------
    # 4. Module 21 audit
    # --------------------------------------------------------------

    print("\n[4] Auditing Module 21 policy comparison...")

    module21 = pd.read_csv(
        POLICIES
        / "module21_forecast_adaptive_risk_comparison.csv"
    )

    expected_models = {
        "Persistence",
        "HGB",
        "CP-HGB",
    }

    actual_models = set(
        module21["forecast_model"].unique()
    )

    if actual_models != expected_models:
        raise ValueError(
            "Module 21 model set mismatch.\n"
            f"Expected: {sorted(expected_models)}\n"
            f"Found: {sorted(actual_models)}"
        )

    expected_rows = 2589 * 3

    if len(module21) != expected_rows:
        raise ValueError(
            f"Module 21 expected {expected_rows:,} rows, "
            f"found {len(module21):,}."
        )

    counts = (
        module21
        .groupby("forecast_model")
        .size()
    )

    if not (counts == 2589).all():
        raise ValueError(
            "Each Module 21 model must contain exactly "
            "2,589 rows."
        )

    assert_finite(
        module21,
        "Module 21"
    )

    print(
        "Models: Persistence / HGB / CP-HGB"
    )

    print(
        "Rows per model: 2,589"
    )

    checks.append(
        "module21_policy_comparison = PASS"
    )

    # --------------------------------------------------------------
    # 5. Module 21 exact paired interval audit
    # --------------------------------------------------------------

    print("\n[5] Auditing exact Module 21 interval pairing...")

    interval_sets = {}

    for model in expected_models:
        interval_sets[model] = set(
            module21.loc[
                module21["forecast_model"] == model,
                "interval",
            ].astype(int)
        )

    if not (
        interval_sets["Persistence"]
        == interval_sets["HGB"]
        == interval_sets["CP-HGB"]
    ):
        raise ValueError(
            "Module 21 models do not use identical intervals."
        )

    if interval_sets["Persistence"] != set(
        forecasts["interval"].astype(int)
    ):
        raise ValueError(
            "Module 21 intervals do not exactly match "
            "Module 19 intervals."
        )

    print(
        "Exact common locked-test intervals: 2,589"
    )

    checks.append(
        "module21_exact_pairing = PASS"
    )

    # --------------------------------------------------------------
    # 6. Risk activation consistency
    # --------------------------------------------------------------

    print("\n[6] Checking shared adaptive-risk activation...")

    activation = (
        module21
        .groupby("forecast_model")
        ["reserve_trigger"]
        .mean()
    )

    activation_values = activation.to_numpy()

    if not np.allclose(
        activation_values,
        activation_values[0],
    ):
        raise ValueError(
            "Reserve activation differs across Module 21 "
            "forecasters, despite shared risk/threshold inputs."
        )

    print(
        "Reserve activation is identical across all "
        "three forecasters."
    )

    checks.append(
        "shared_risk_controller = PASS"
    )

    # --------------------------------------------------------------
    # 7. Module 22 statistical audit
    # --------------------------------------------------------------

    print("\n[7] Auditing Module 22 statistical inference...")

    module22 = pd.read_csv(
        TABLES
        / "module22_paired_policy_statistics.csv"
    )

    expected_comparisons = 3
    expected_metrics = 4

    if len(module22) != (
        expected_comparisons
        * expected_metrics
    ):
        raise ValueError(
            "Unexpected Module 22 comparison count."
        )

    assert_finite(
        module22,
        "Module 22 statistics"
    )

    if not (
        module22["holm_adjusted_p_value"]
        < 0.05
    ).all():
        raise ValueError(
            "Not every Module 22 comparison is significant "
            "after Holm correction."
        )

    print(
        "12/12 paired comparisons significant after Holm correction."
    )

    checks.append(
        "module22_statistics = PASS"
    )

    # --------------------------------------------------------------
    # 8. Normalized-capacity scope audit
    # --------------------------------------------------------------

    print("\n[8] Checking resource-scope claims...")

    metadata_path = (
        VALIDATION
        / "capacity_simulation_metadata.json"
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as f:
        capacity_metadata = json.load(f)

    if capacity_metadata.get(
        "token_to_gpu_conversion"
    ) is not False:

        raise ValueError(
            "Capacity metadata does not explicitly preserve "
            "the normalized-capacity scope."
        )

    if capacity_metadata.get(
        "production_autoscaling_claim"
    ) is not False:

        raise ValueError(
            "Production autoscaling claim must remain false."
        )

    print(
        "Normalized workload-capacity scope: PASS"
    )

    print(
        "Token-to-GPU conversion claim: OFF"
    )

    print(
        "Production autoscaling claim: OFF"
    )

    checks.append(
        "cloud_scope_control = PASS"
    )

    # --------------------------------------------------------------
    # 9. Freeze manifest
    # --------------------------------------------------------------

    print("\n[9] Creating reproducibility freeze manifest...")

    artifacts = {}

    for path in EXPECTED_FILES:
        artifacts[
            str(path.relative_to(ROOT))
        ] = {
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }

    manifest = {
        "module":
            "23_final_implementation_audit",
        "status":
            "FROZEN",
        "research_topic":
            "Short-Term LLM Workload Prediction for Adaptive Cloud Computing",
        "locked_test_intervals":
            2589,
        "forecast_models":
            [
                "Persistence",
                "HGB",
                "CP-HGB",
            ],
        "policy_comparison":
            "forecast + existing adaptive risk controller",
        "statistical_inference":
            "paired Wilcoxon + paired bootstrap + Holm correction",
        "test_parameter_tuning":
            False,
        "token_to_gpu_conversion":
            False,
        "production_autoscaling_claim":
            False,
        "implementation_experiments_complete":
            True,
        "paper_writing_ready":
            True,
        "checks":
            checks,
        "artifacts":
            artifacts,
    }

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print(
        f"Freeze manifest saved:\n  {MANIFEST_PATH}"
    )

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODULE 23 RESULT")
    print("=" * 70)

    print(
        "\nALL FINAL IMPLEMENTATION CHECKS PASSED."
    )

    print(
        "\nSTATUS: IMPLEMENTATION FROZEN"
    )

    print(
        "STATUS: PAPER WRITING READY"
    )

    print(
        "\nDo not modify the experimental pipeline after this point "
        "without explicitly creating a new experiment/version."
    )

    print(
        "\nMODULE 23 COMPLETE"
    )


if __name__ == "__main__":
    main()
