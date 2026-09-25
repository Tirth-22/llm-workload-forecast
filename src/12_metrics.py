import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
)


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

RISK_PREDICTIONS_PATH = (
    BASE_DIR
    / "results"
    / "predictions"
    / "calibrated_failure_risk_predictions.csv"
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

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "tables"
)

OUTPUT_JSON = (
    RESULTS_DIR
    / "evaluation_metrics.json"
)

OUTPUT_FORECAST = (
    RESULTS_DIR
    / "forecast_metrics.csv"
)

OUTPUT_RISK = (
    RESULTS_DIR
    / "risk_metrics.csv"
)

OUTPUT_CAPACITY = (
    RESULTS_DIR
    / "capacity_metrics.csv"
)


# ======================================================================
# CONTROLLER CONFIGURATION
# ======================================================================

# Paper 4 validation-selected threshold.
# This is NOT tuned on the locked test set.
THETA_BASE = 0.9591


# ======================================================================
# FORECAST METRICS
# ======================================================================

def mae(actual, predicted):

    return float(
        np.mean(
            np.abs(
                actual - predicted
            )
        )
    )


def rmse(actual, predicted):

    return float(
        np.sqrt(
            np.mean(
                (actual - predicted) ** 2
            )
        )
    )


def wape(actual, predicted):

    denominator = np.sum(
        np.abs(actual)
    )

    if denominator == 0:
        return np.nan

    return float(
        np.sum(
            np.abs(
                actual - predicted
            )
        )
        / denominator
        * 100
    )


def smape(actual, predicted):

    denominator = (
        np.abs(actual)
        +
        np.abs(predicted)
    )

    valid = denominator > 0

    if not np.any(valid):
        return np.nan

    return float(
        np.mean(
            2
            * np.abs(
                actual[valid]
                -
                predicted[valid]
            )
            /
            denominator[valid]
        )
        * 100
    )


def mean_bias(actual, predicted):

    return float(
        np.mean(
            predicted - actual
        )
    )


# ======================================================================
# ECE
# ======================================================================

def calculate_ece(
    y_true,
    probabilities,
    bins=10
):

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    ece = 0.0

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1
    )

    for i in range(bins):

        lower = edges[i]
        upper = edges[i + 1]

        if i == bins - 1:

            mask = (
                (probabilities >= lower)
                &
                (probabilities <= upper)
            )

        else:

            mask = (
                (probabilities >= lower)
                &
                (probabilities < upper)
            )

        if not np.any(mask):
            continue

        confidence = np.mean(
            probabilities[mask]
        )

        accuracy = np.mean(
            y_true[mask]
        )

        fraction = np.mean(mask)

        ece += (
            fraction
            *
            abs(
                confidence
                -
                accuracy
            )
        )

    return float(ece)


# ======================================================================
# CAPACITY METRICS
# ======================================================================

def capacity_metrics(
    df,
    actual_column,
    provisioned_column,
    reserve_column
):

    actual = (
        df[actual_column]
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
        .astype(int)
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

    p95 = float(
        np.percentile(
            shortage,
            95
        )
    )

    tail = shortage[
        shortage >= p95
    ]

    cvar95 = (
        float(np.mean(tail))
        if len(tail) > 0
        else 0.0
    )

    mean_actual = float(
        np.mean(actual)
    )

    mean_provisioned = float(
        np.mean(provisioned)
    )

    if mean_actual > 0:

        cost_index = (
            mean_provisioned
            /
            mean_actual
        )

    else:

        cost_index = np.nan

    return {

        "intervals":
            int(len(df)),

        "under_provisioned_percent":
            float(
                np.mean(
                    shortage > 0
                )
                * 100
            ),

        "mean_shortage":
            float(
                np.mean(shortage)
            ),

        "p95_shortage":
            p95,

        "shortfall_cvar95":
            cvar95,

        "mean_unused_capacity":
            float(
                np.mean(unused)
            ),

        "mean_provisioned_capacity":
            mean_provisioned,

        "mean_actual_workload":
            mean_actual,

        "reserve_activation_percent":
            float(
                np.mean(reserve)
                * 100
            ),

        "cost_index":
            float(cost_index),

        "total_shortage":
            float(
                np.sum(shortage)
            ),

        "total_unused_capacity":
            float(
                np.sum(unused)
            )
    }


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("MODULE 12 — COMPREHENSIVE EVALUATION")
    print("=" * 70)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = {}

    # ==================================================================
    # PART A — WORKLOAD FORECASTING
    # ==================================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PART A — WORKLOAD FORECASTING"
    )

    print(
        "=" * 70
    )

    if not WORKLOAD_PREDICTIONS_PATH.exists():

        raise FileNotFoundError(
            f"Workload predictions not found:\n"
            f"{WORKLOAD_PREDICTIONS_PATH}"
        )

    forecast = pd.read_csv(
        WORKLOAD_PREDICTIONS_PATH
    )

    print(
        f"Rows loaded: {len(forecast):,}"
    )

    print(
        f"Columns: {list(forecast.columns)}"
    )

    test_forecast = forecast[
        forecast["split"]
        .astype(str)
        .str.lower()
        == "test"
    ].copy()

    if len(test_forecast) == 0:

        raise ValueError(
            "No workload test rows found."
        )

    print(
        f"Locked test forecast rows: "
        f"{len(test_forecast):,}"
    )

    required_forecast = [
        "actual_next_5min_tokens",
        "hgb_prediction",
        "cp_hgb_prediction"
    ]

    missing = [
        c
        for c in required_forecast
        if c not in test_forecast.columns
    ]

    if missing:

        raise ValueError(
            "Missing workload columns:\n"
            + "\n".join(missing)
        )

    actual = (
        test_forecast[
            "actual_next_5min_tokens"
        ]
        .astype(float)
        .to_numpy()
    )

    hgb_pred = (
        test_forecast[
            "hgb_prediction"
        ]
        .astype(float)
        .to_numpy()
    )

    cp_hgb_pred = (
        test_forecast[
            "cp_hgb_prediction"
        ]
        .astype(float)
        .to_numpy()
    )

    forecast_rows = []

    for model_name, prediction in [
        ("HGB", hgb_pred),
        ("CP-HGB", cp_hgb_pred)
    ]:

        metrics = {

            "model":
                model_name,

            "test_rows":
                int(len(actual)),

            "MAE":
                mae(
                    actual,
                    prediction
                ),

            "RMSE":
                rmse(
                    actual,
                    prediction
                ),

            "WAPE_percent":
                wape(
                    actual,
                    prediction
                ),

            "sMAPE_percent":
                smape(
                    actual,
                    prediction
                ),

            "mean_bias":
                mean_bias(
                    actual,
                    prediction
                )
        }

        forecast_rows.append(
            metrics
        )

        print(
            f"\n{model_name}"
        )

        print(
            f"  MAE: "
            f"{metrics['MAE']:.4f}"
        )

        print(
            f"  RMSE: "
            f"{metrics['RMSE']:.4f}"
        )

        print(
            f"  WAPE: "
            f"{metrics['WAPE_percent']:.4f}%"
        )

        print(
            f"  sMAPE: "
            f"{metrics['sMAPE_percent']:.4f}%"
        )

        print(
            f"  Mean bias: "
            f"{metrics['mean_bias']:.4f}"
        )

    forecast_df = pd.DataFrame(
        forecast_rows
    )

    forecast_df.to_csv(
        OUTPUT_FORECAST,
        index=False
    )

    all_results["forecast"] = forecast_rows

    # ==================================================================
    # PART B — FAILURE RISK
    # ==================================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PART B — FAILURE-RISK EVALUATION"
    )

    print(
        "=" * 70
    )

    if not RISK_PREDICTIONS_PATH.exists():

        raise FileNotFoundError(
            f"Risk predictions not found:\n"
            f"{RISK_PREDICTIONS_PATH}"
        )

    risk = pd.read_csv(
        RISK_PREDICTIONS_PATH
    )

    print(
        f"Rows loaded: {len(risk):,}"
    )

    print(
        f"Columns: {list(risk.columns)}"
    )

    required_risk = [
        "split",
        "failure",
        "predicted_failure_probability",
        "calibrated_failure_probability"
    ]

    missing = [
        c
        for c in required_risk
        if c not in risk.columns
    ]

    if missing:

        raise ValueError(
            "Missing risk columns:\n"
            + "\n".join(missing)
        )

    test_risk = risk[
        risk["split"]
        .astype(str)
        .str.lower()
        == "test"
    ].copy()

    if len(test_risk) == 0:

        raise ValueError(
            "No risk test rows found."
        )

    print(
        f"Locked test risk rows: "
        f"{len(test_risk):,}"
    )

    y_true = (
        test_risk[
            "failure"
        ]
        .astype(int)
        .to_numpy()
    )

    # --------------------------------------------------------------
    # IMPORTANT:
    #
    # Formal probability evaluation uses the calibrated probabilities
    # generated by Module 08.
    # --------------------------------------------------------------

    probability = (
        test_risk[
            "calibrated_failure_probability"
        ]
        .astype(float)
        .to_numpy()
    )

    raw_probability = (
        test_risk[
            "predicted_failure_probability"
        ]
        .astype(float)
        .to_numpy()
    )

    if (
        np.min(probability) < 0
        or
        np.max(probability) > 1
    ):

        raise ValueError(
            "Calibrated risk probabilities outside [0,1]."
        )

    prevalence = float(
        np.mean(y_true)
    )

    # --------------------------------------------------------------
    # Calibrated probability metrics
    # --------------------------------------------------------------

    roc_auc = roc_auc_score(
        y_true,
        probability
    )

    average_precision = (
        average_precision_score(
            y_true,
            probability
        )
    )

    brier = brier_score_loss(
        y_true,
        probability
    )

    ece = calculate_ece(
        y_true,
        probability,
        bins=10
    )

    # --------------------------------------------------------------
    # Raw probability diagnostics
    # --------------------------------------------------------------

    raw_brier = brier_score_loss(
        y_true,
        raw_probability
    )

    # --------------------------------------------------------------
    # Classification threshold
    #
    # This is the documented Paper 4 validation-selected base
    # threshold. It is NOT optimized using test data.
    # --------------------------------------------------------------

    classification_threshold = THETA_BASE

    predicted_class = (
        probability
        >
        classification_threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predicted_class,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predicted_class,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predicted_class,
        zero_division=0
    )

    risk_metrics = {

        "test_rows":
            int(len(test_risk)),

        "failure_prevalence":
            prevalence,

        "failure_prevalence_percent":
            prevalence * 100,

        "ROC_AUC":
            float(roc_auc),

        "Average_Precision":
            float(average_precision),

        "Brier_calibrated":
            float(brier),

        "Brier_raw_diagnostic":
            float(raw_brier),

        "ECE_10_bin":
            float(ece),

        "classification_threshold":
            classification_threshold,

        "Precision":
            float(precision),

        "Recall":
            float(recall),

        "F1":
            float(f1)
    }

    print(
        f"\nFailure prevalence: "
        f"{prevalence * 100:.4f}%"
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc:.6f}"
    )

    print(
        f"Average Precision: "
        f"{average_precision:.6f}"
    )

    print(
        f"Calibrated Brier: "
        f"{brier:.6f}"
    )

    print(
        f"Raw Brier diagnostic: "
        f"{raw_brier:.6f}"
    )

    print(
        f"10-bin ECE: "
        f"{ece:.6f}"
    )

    print(
        f"Precision @ {classification_threshold:.4f}: "
        f"{precision:.6f}"
    )

    print(
        f"Recall @ {classification_threshold:.4f}: "
        f"{recall:.6f}"
    )

    print(
        f"F1 @ {classification_threshold:.4f}: "
        f"{f1:.6f}"
    )

    pd.DataFrame(
        [risk_metrics]
    ).to_csv(
        OUTPUT_RISK,
        index=False
    )

    all_results["risk"] = risk_metrics

    # ==================================================================
    # PART C — CAPACITY POLICIES
    # ==================================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PART C — CAPACITY POLICY EVALUATION"
    )

    print(
        "=" * 70
    )

    policy_definitions = {

        "V1": (
            V1_PATH,
            "V1_reserve_trigger",
            "V1_provisioned_capacity"
        ),

        "V2": (
            V2_PATH,
            "V2_reserve_trigger",
            "V2_provisioned_capacity"
        ),

        "V3": (
            V3_PATH,
            "V3_reserve_trigger",
            "V3_provisioned_capacity"
        ),

        "V4": (
            V4_PATH,
            "V4_reserve_trigger",
            "V4_provisioned_capacity"
        )
    }

    capacity_rows = []

    for policy_name, (
        path,
        reserve_column,
        capacity_column
    ) in policy_definitions.items():

        print(
            f"\nEvaluating {policy_name}..."
        )

        if not path.exists():

            raise FileNotFoundError(
                f"{policy_name} file not found:\n"
                f"{path}"
            )

        df = pd.read_csv(
            path
        )

        if "split" not in df.columns:

            raise ValueError(
                f"{policy_name} missing split column."
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

        required = [
            "actual_workload",
            reserve_column,
            capacity_column
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

        metrics = capacity_metrics(
            df,
            "actual_workload",
            capacity_column,
            reserve_column
        )

        metrics["policy"] = policy_name

        capacity_rows.append(
            metrics
        )

        print(
            f"  Test intervals: "
            f"{metrics['intervals']:,}"
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
            f"  P95 shortage: "
            f"{metrics['p95_shortage']:.4f}"
        )

        print(
            f"  CVaR95 shortage: "
            f"{metrics['shortfall_cvar95']:.4f}"
        )

        print(
            f"  Mean unused capacity: "
            f"{metrics['mean_unused_capacity']:.4f}"
        )

        print(
            f"  Mean provisioned capacity: "
            f"{metrics['mean_provisioned_capacity']:.4f}"
        )

        print(
            f"  Reserve activation: "
            f"{metrics['reserve_activation_percent']:.4f}%"
        )

        print(
            f"  Cost index: "
            f"{metrics['cost_index']:.6f}"
        )

    capacity_df = pd.DataFrame(
        capacity_rows
    )

    capacity_columns = [
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
        "cost_index",
        "total_shortage",
        "total_unused_capacity"
    ]

    capacity_df = capacity_df[
        capacity_columns
    ]

    capacity_df.to_csv(
        OUTPUT_CAPACITY,
        index=False
    )

    all_results["capacity"] = capacity_rows

    # ==================================================================
    # PART D — SAVE COMPLETE RESULTS
    # ==================================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SAVING COMPLETE EVALUATION"
    )

    print(
        "=" * 70
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_results,
            file,
            indent=2
        )

    print(
        "\nCreated:"
    )

    print(
        "  results/tables/"
        "forecast_metrics.csv"
    )

    print(
        "  results/tables/"
        "risk_metrics.csv"
    )

    print(
        "  results/tables/"
        "capacity_metrics.csv"
    )

    print(
        "  results/tables/"
        "evaluation_metrics.json"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODULE 12 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nNo model was retrained."
    )

    print(
        "No test threshold was optimized."
    )

    print(
        "No policy was declared better."
    )

    print(
        "Locked test data was used only for evaluation."
    )


if __name__ == "__main__":
    main()