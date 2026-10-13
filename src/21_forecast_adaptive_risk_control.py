"""
Module 21 — Forecast + Adaptive Risk Control Comparison

Research topic:
Short-Term LLM Workload Prediction for Adaptive Cloud Computing

Purpose
-------
Compare Persistence, HGB, and CP-HGB inside the EXISTING adaptive
failure-risk controller.

The experiment isolates the effect of the workload forecast while keeping
the failure-risk model and adaptive threshold fixed.

Models compared:
    Persistence
    HGB
    CP-HGB

Existing risk outputs are reused from Module 08 / existing prediction file.

Important:
- No risk model is retrained.
- No threshold is tuned on the locked test set.
- No new change-point detector is introduced.
- The existing validation-selected base threshold is retained.
- The forecast model is the only changed component.
- This module is an attribution experiment, not a claim that one model
  is universally superior.

Inputs
------
results/predictions/baseline_forecast_predictions.csv
results/predictions/calibrated_failure_risk_predictions.csv

Outputs
-------
results/policies/module21_forecast_adaptive_risk_comparison.csv
results/tables/module21_forecast_adaptive_risk_metrics.csv
results/validation/module21_metadata.json
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FORECAST_INPUT = (
    ROOT
    / "results"
    / "predictions"
    / "baseline_forecast_predictions.csv"
)

RISK_INPUT = (
    ROOT
    / "results"
    / "predictions"
    / "calibrated_failure_risk_predictions.csv"
)

POLICY_DIR = ROOT / "results" / "policies"
TABLE_DIR = ROOT / "results" / "tables"
VALIDATION_DIR = ROOT / "results" / "validation"

POLICY_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# EXISTING CONTROLLER SETTINGS
# ----------------------------------------------------------------------

# Validation-selected base threshold from the existing implementation.
# It is NOT tuned on the locked test set.
BASE_THRESHOLD = 0.9591

# Existing adaptive-threshold parameters documented by the project.
ETA = 0.01
MIN_THRESHOLD = 0.50
MAX_THRESHOLD = 0.99
HISTORICAL_WINDOW = 60

# Existing calibration-derived target failure rate.
TARGET_FAILURE_RATE = 0.044893

# Existing reserve policy.
RESERVE_FRACTION = 0.10


# ----------------------------------------------------------------------
# ACTUAL MODULE 19 FORECAST COLUMNS
# ----------------------------------------------------------------------

FORECAST_COLUMNS = {
    "Persistence": "persistence_prediction",
    "HGB": "hgb_prediction",
    "CP-HGB": "cp_hgb_prediction",
}


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def find_column(df, candidates, description):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Could not resolve {description}. Checked:\n"
        + "\n".join(f"  - {x}" for x in candidates)
    )


def load_forecasts():
    if not FORECAST_INPUT.exists():
        raise FileNotFoundError(
            f"Module 19 predictions not found:\n{FORECAST_INPUT}"
        )

    df = pd.read_csv(FORECAST_INPUT)

    required = {
        "interval",
        "actual_next_5min_tokens",
        *FORECAST_COLUMNS.values(),
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Module 19 output is missing:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    df = df.sort_values("interval").reset_index(drop=True)

    if df.empty:
        raise ValueError("Module 19 prediction file is empty.")

    return df


def load_risk():
    if not RISK_INPUT.exists():
        raise FileNotFoundError(
            f"Calibrated risk predictions not found:\\n{RISK_INPUT}"
        )

    risk = pd.read_csv(RISK_INPUT)

    required = {"minute", "split", "failure", "calibrated_failure_probability"}
    missing = sorted(required - set(risk.columns))
    if missing:
        raise ValueError(
            "Risk prediction file is missing:\\n"
            + "\\n".join(f"  - {x}" for x in missing)
        )

    risk["minute"] = risk["minute"].astype(int)
    risk["interval"] = risk["minute"] // 5

    # EXACT aggregation used by Module 10:
    # maximum calibrated minute-level risk within each 5-minute interval.
    risk_5min = (
        risk.groupby("interval", as_index=False)
        .agg(risk_signal=("calibrated_failure_probability", "max"))
    )

    return risk_5min


def load_adaptive_thresholds():
    path = ROOT / "results" / "policies" / "adaptive_threshold_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Adaptive threshold output not found:\\n{path}")

    adaptive = pd.read_csv(path)
    required = {"minute", "split", "threshold_before_update"}
    missing = sorted(required - set(adaptive.columns))
    if missing:
        raise ValueError(
            "Adaptive threshold file is missing:\\n"
            + "\\n".join(f"  - {x}" for x in missing)
        )

    adaptive["minute"] = adaptive["minute"].astype(int)
    adaptive["interval"] = adaptive["minute"] // 5

    # EXACT aggregation used by Module 10:
    # first threshold available within each five-minute interval.
    adaptive_5min = (
        adaptive.sort_values("minute")
        .groupby("interval", as_index=False)
        .first()
    )

    return adaptive_5min[["interval", "threshold_before_update"]]


def evaluate_controller(actual, forecast, risk_probability, adaptive_threshold):
    # EXACT decision rule used by Module 10 V3:
    # risk_signal > adaptive_threshold -> reserve trigger.
    reserve_trigger = (
        risk_probability > adaptive_threshold
    ).astype(int)

    provisioned = forecast * np.where(
        reserve_trigger == 1,
        RESERVE_FRACTION,
        1.0,
    )

    shortage = np.maximum(actual - provisioned, 0.0)
    unused = np.maximum(provisioned - actual, 0.0)

    p95 = float(np.percentile(shortage, 95))
    tail = shortage[shortage >= p95]
    cvar95 = float(np.mean(tail)) if len(tail) else 0.0
    mean_actual = float(np.mean(actual))
    mean_provisioned = float(np.mean(provisioned))

    return {
        "under_provisioned_percent": float(np.mean(shortage > 0) * 100),
        "mean_shortage": float(np.mean(shortage)),
        "p95_shortage": p95,
        "cvar95_shortage": cvar95,
        "mean_unused_capacity": float(np.mean(unused)),
        "mean_provisioned_capacity": mean_provisioned,
        "mean_actual_workload": mean_actual,
        "cost_index": mean_provisioned / mean_actual if mean_actual > 0 else np.nan,
        "reserve_activation_percent": float(np.mean(reserve_trigger) * 100),
        "mean_adaptive_threshold": float(np.mean(adaptive_threshold)),
        "min_adaptive_threshold": float(np.min(adaptive_threshold)),
        "max_adaptive_threshold": float(np.max(adaptive_threshold)),
        "total_shortage": float(np.sum(shortage)),
        "total_unused_capacity": float(np.sum(unused)),
    }, reserve_trigger, provisioned, shortage, unused

def main():
    print("=" * 70)
    print("MODULE 21 — FORECAST + ADAPTIVE RISK CONTROL")
    print("=" * 70)

    print("\\n[1] Loading Module 19 forecasts...")
    forecast_df = load_forecasts()
    print(f"Forecast rows: {len(forecast_df):,}")
    print(f"Forecast interval range: {int(forecast_df['interval'].min())} -> {int(forecast_df['interval'].max())}")

    print("\\n[2] Loading existing minute-level calibrated risk...")
    risk_5min = load_risk()
    print(f"Five-minute risk intervals: {len(risk_5min):,}")

    print("\\n[3] Loading existing Module 09 adaptive thresholds...")
    adaptive_5min = load_adaptive_thresholds()
    print(f"Five-minute threshold intervals: {len(adaptive_5min):,}")

    print("\\n[4] Reusing EXACT Module 10 aggregation rules...")
    print("  Risk aggregation       : maximum calibrated risk per 5-min interval")
    print("  Threshold aggregation : first threshold in each 5-min interval")
    print("  Reserve rule           : risk_signal > adaptive_threshold")
    print("  Reserve factor         : 1.10")

    merged = forecast_df.merge(risk_5min, on="interval", how="inner", validate="one_to_one")
    merged = merged.merge(adaptive_5min, on="interval", how="inner", validate="one_to_one")
    merged = merged.sort_values("interval").reset_index(drop=True)

    if len(merged) != len(forecast_df):
        raise ValueError(
            f"Locked-test forecast rows not fully aligned with risk/threshold views: "
            f"forecast={len(forecast_df)}, merged={len(merged)}"
        )

    print("\\n[5] Exact locked-test alignment: PASS")
    print(f"Aligned rows: {len(merged):,}")

    actual = merged["actual_next_5min_tokens"].astype(float).to_numpy()
    risk_probability = merged["risk_signal"].astype(float).to_numpy()
    adaptive_threshold = merged["threshold_before_update"].astype(float).to_numpy()

    if not np.isfinite(actual).all():
        raise ValueError("Actual workload contains non-finite values.")
    if not np.isfinite(risk_probability).all() or not ((risk_probability >= 0).all() and (risk_probability <= 1).all()):
        raise ValueError("Risk signal outside [0,1] or non-finite.")
    if not np.isfinite(adaptive_threshold).all():
        raise ValueError("Adaptive thresholds contain non-finite values.")

    metrics_rows = []
    policy_rows = []

    print("\\n[6] Running the SAME existing adaptive-risk controller for all forecasts...")

    for model_name, forecast_column in FORECAST_COLUMNS.items():
        forecast = merged[forecast_column].astype(float).clip(lower=0).to_numpy()

        metrics, triggers, provisioned, shortage, unused = evaluate_controller(
            actual, forecast, risk_probability, adaptive_threshold
        )
        metrics["forecast_model"] = model_name
        metrics["forecast_column"] = forecast_column
        metrics_rows.append(metrics)

        for i in range(len(merged)):
            policy_rows.append({
                "interval": int(merged.loc[i, "interval"]),
                "forecast_model": model_name,
                "actual_workload": float(actual[i]),
                "forecast_workload": float(forecast[i]),
                "risk_signal": float(risk_probability[i]),
                "adaptive_threshold": float(adaptive_threshold[i]),
                "reserve_trigger": int(triggers[i]),
                "reserve_factor": RESERVE_FRACTION,
                "provisioned_capacity": float(provisioned[i]),
                "shortage": float(shortage[i]),
                "unused_capacity": float(unused[i]),
            })

        print(f"\\n{model_name}")
        print(f"  Under-provisioned: {metrics['under_provisioned_percent']:.4f}%")
        print(f"  Mean shortage: {metrics['mean_shortage']:.4f}")
        print(f"  P95 shortage: {metrics['p95_shortage']:.4f}")
        print(f"  CVaR95 shortage: {metrics['cvar95_shortage']:.4f}")
        print(f"  Mean unused capacity: {metrics['mean_unused_capacity']:.4f}")
        print(f"  Mean provisioned capacity: {metrics['mean_provisioned_capacity']:.4f}")
        print(f"  Cost index: {metrics['cost_index']:.6f}")
        print(f"  Reserve activation: {metrics['reserve_activation_percent']:.4f}%")

    metrics_df = pd.DataFrame(metrics_rows)[[
        "forecast_model", "forecast_column", "under_provisioned_percent",
        "mean_shortage", "p95_shortage", "cvar95_shortage",
        "mean_unused_capacity", "mean_provisioned_capacity",
        "mean_actual_workload", "cost_index", "reserve_activation_percent",
        "mean_adaptive_threshold", "min_adaptive_threshold", "max_adaptive_threshold",
        "total_shortage", "total_unused_capacity",
    ]]
    policy_df = pd.DataFrame(policy_rows)

    metrics_path = TABLE_DIR / "module21_forecast_adaptive_risk_metrics.csv"
    policy_path = POLICY_DIR / "module21_forecast_adaptive_risk_comparison.csv"
    metadata_path = VALIDATION_DIR / "module21_metadata.json"

    metrics_df.to_csv(metrics_path, index=False)
    policy_df.to_csv(policy_path, index=False)

    metadata = {
        "module": "21_forecast_adaptive_risk_control",
        "research_topic": "Short-Term LLM Workload Prediction for Adaptive Cloud Computing",
        "forecast_input": str(FORECAST_INPUT),
        "risk_input": str(RISK_INPUT),
        "adaptive_threshold_input": str(ROOT / "results" / "policies" / "adaptive_threshold_predictions.csv"),
        "forecast_models": ["Persistence", "HGB", "CP-HGB"],
        "risk_model_retrained": False,
        "risk_aggregation": "maximum calibrated minute-level risk within each five-minute interval (exact Module 10 rule)",
        "adaptive_threshold_aggregation": "first threshold available within each five-minute interval (exact Module 10 rule)",
        "reserve_rule": "risk_signal > adaptive_threshold",
        "reserve_factor": RESERVE_FRACTION,
        "test_set_parameter_tuning": False,
        "forecast_models_retrained": False,
        "purpose": "Isolate the effect of the short-term workload forecaster while holding the existing calibrated-risk adaptive controller fixed.",
        "scope_note": "Capacity remains normalized workload capacity; no direct token-to-GPU conversion is claimed.",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\\n" + "=" * 70)
    print("MODULE 21 RESULT")
    print("=" * 70)
    print(metrics_df[[
        "forecast_model", "under_provisioned_percent", "mean_shortage",
        "p95_shortage", "cvar95_shortage", "mean_unused_capacity",
        "mean_provisioned_capacity", "cost_index", "reserve_activation_percent"
    ]].to_string(index=False))
    print("\\nCreated:")
    print(f"  {policy_path}")
    print(f"  {metrics_path}")
    print(f"  {metadata_path}")
    print("\\nExisting risk model and adaptive threshold state were reused unchanged.")
    print("No test parameter was tuned.")
    print("MODULE 21 COMPLETE")


if __name__ == "__main__":
    main()
