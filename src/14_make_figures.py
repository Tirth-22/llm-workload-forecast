import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================
# PATHS
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PREDICTIONS_DIR = (
    BASE_DIR
    / "results"
    / "predictions"
)

ALARMS_PATH = (
    BASE_DIR
    / "results"
    / "alarms"
    / "change_point_5min.csv"
)

POLICIES_DIR = (
    BASE_DIR
    / "results"
    / "policies"
)

TABLES_DIR = (
    BASE_DIR
    / "results"
    / "tables"
)

FIGURES_DIR = (
    BASE_DIR
    / "results"
    / "figures"
)

WORKLOAD_PATH = (
    PREDICTIONS_DIR
    / "workload_forecast_predictions.csv"
)

RISK_PATH = (
    PREDICTIONS_DIR
    / "calibrated_failure_risk_predictions.csv"
)

FORECAST_METRICS_PATH = (
    TABLES_DIR
    / "forecast_metrics.csv"
)

RISK_METRICS_PATH = (
    TABLES_DIR
    / "risk_metrics.csv"
)

CAPACITY_METRICS_PATH = (
    TABLES_DIR
    / "capacity_metrics.csv"
)

BOOTSTRAP_PATH = (
    TABLES_DIR
    / "bootstrap_tests.csv"
)

V1_PATH = (
    POLICIES_DIR
    / "V1_forecast_only.csv"
)

V2_PATH = (
    POLICIES_DIR
    / "V2_fixed_risk.csv"
)

V3_PATH = (
    POLICIES_DIR
    / "V3_adaptive_risk.csv"
)

V4_PATH = (
    POLICIES_DIR
    / "V4_adaptive_reset.csv"
)


# ======================================================================
# FIGURE CONFIGURATION
# ======================================================================

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DPI = 300


# ======================================================================
# HELPERS
# ======================================================================

def save_figure(
    filename
):
    """
    Save the current matplotlib figure.
    """

    path = (
        FIGURES_DIR
        / filename
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Created: results/figures/{filename}"
    )


def load_test_workload():

    df = pd.read_csv(
        WORKLOAD_PATH
    )

    df = df[
        df["split"]
        .astype(str)
        .str.lower()
        == "test"
    ].copy()

    return (
        df
        .sort_values("interval")
        .reset_index(drop=True)
    )


def load_test_risk():

    df = pd.read_csv(
        RISK_PATH
    )

    df = df[
        df["split"]
        .astype(str)
        .str.lower()
        == "test"
    ].copy()

    return (
        df
        .sort_values("minute")
        .reset_index(drop=True)
    )


def load_policy(
    path
):

    df = pd.read_csv(
        path
    )

    df = df[
        df["split"]
        .astype(str)
        .str.lower()
        == "test"
    ].copy()

    return (
        df
        .sort_values("interval")
        .reset_index(drop=True)
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("MODULE 14 — RESEARCH FIGURES")
    print("=" * 70)

    # ==================================================================
    # VALIDATE REQUIRED FILES
    # ==================================================================

    required_files = [
        WORKLOAD_PATH,
        RISK_PATH,
        ALARMS_PATH,
        FORECAST_METRICS_PATH,
        RISK_METRICS_PATH,
        CAPACITY_METRICS_PATH,
        BOOTSTRAP_PATH,
        V1_PATH,
        V2_PATH,
        V3_PATH,
        V4_PATH
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # ==================================================================
    # LOAD DATA
    # ==================================================================

    print("\nLoading evaluation data...")

    workload = load_test_workload()

    risk = load_test_risk()

    alarms = pd.read_csv(
        ALARMS_PATH
    )

    forecast_metrics = pd.read_csv(
        FORECAST_METRICS_PATH
    )

    risk_metrics = pd.read_csv(
        RISK_METRICS_PATH
    )

    capacity_metrics = pd.read_csv(
        CAPACITY_METRICS_PATH
    )

    bootstrap = pd.read_csv(
        BOOTSTRAP_PATH
    )

    v1 = load_policy(V1_PATH)
    v2 = load_policy(V2_PATH)
    v3 = load_policy(V3_PATH)
    v4 = load_policy(V4_PATH)

    print(
        f"Forecast test rows: {len(workload):,}"
    )

    print(
        f"Risk test rows: {len(risk):,}"
    )

    print(
        f"Change-point rows: {len(alarms):,}"
    )

    # ==================================================================
    # FIGURE 1
    # FORECAST ERROR METRICS
    # ==================================================================

    print(
        "\n[1] Forecast metric comparison..."
    )

    metrics_plot = forecast_metrics[
        [
            "model",
            "MAE",
            "RMSE"
        ]
    ].copy()

    x = np.arange(
        len(metrics_plot)
    )

    width = 0.35

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        x - width / 2,
        metrics_plot["MAE"],
        width,
        label="MAE"
    )

    plt.bar(
        x + width / 2,
        metrics_plot["RMSE"],
        width,
        label="RMSE"
    )

    plt.xticks(
        x,
        metrics_plot["model"]
    )

    plt.ylabel(
        "Error (tokens / 5-min interval)"
    )

    plt.xlabel(
        "Forecast model"
    )

    plt.title(
        "Workload Forecast Error on Locked Test Set"
    )

    plt.legend()

    save_figure(
        "fig01_forecast_error_comparison.png"
    )

    # ==================================================================
    # FIGURE 2
    # ACTUAL VS PREDICTED WORKLOAD
    # ==================================================================

    print(
        "\n[2] Actual vs predicted workload..."
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        workload["interval"],
        workload["actual_next_5min_tokens"],
        label="Actual workload",
        linewidth=1.0
    )

    plt.plot(
        workload["interval"],
        workload["hgb_prediction"],
        label="HGB",
        linewidth=0.9
    )

    plt.plot(
        workload["interval"],
        workload["cp_hgb_prediction"],
        label="CP-HGB",
        linewidth=0.9
    )

    plt.xlabel(
        "Five-minute interval"
    )

    plt.ylabel(
        "Response tokens"
    )

    plt.title(
        "Actual and Forecast LLM Workload — Locked Test Period"
    )

    plt.legend()

    save_figure(
        "fig02_actual_vs_forecast.png"
    )

    # ==================================================================
    # FIGURE 3
    # ABSOLUTE FORECAST ERROR
    # ==================================================================

    print(
        "\n[3] Forecast absolute error..."
    )

    hgb_error = np.abs(
        workload["actual_next_5min_tokens"]
        -
        workload["hgb_prediction"]
    )

    cp_hgb_error = np.abs(
        workload["actual_next_5min_tokens"]
        -
        workload["cp_hgb_prediction"]
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        workload["interval"],
        hgb_error,
        label="HGB absolute error",
        linewidth=0.8
    )

    plt.plot(
        workload["interval"],
        cp_hgb_error,
        label="CP-HGB absolute error",
        linewidth=0.8
    )

    plt.xlabel(
        "Five-minute interval"
    )

    plt.ylabel(
        "Absolute error (tokens)"
    )

    plt.title(
        "Forecast Absolute Error Over the Locked Test Period"
    )

    plt.legend()

    save_figure(
        "fig03_forecast_absolute_error.png"
    )

    # ==================================================================
    # FIGURE 4
    # CHANGE-POINT ALARMS
    # ==================================================================

    print(
        "\n[4] Change-point alarms..."
    )

    test_start = int(
        workload["interval"].min()
    )

    test_end = int(
        workload["interval"].max()
    )

    alarm_test = alarms[
        (alarms["interval"] >= test_start)
        &
        (alarms["interval"] <= test_end)
    ].copy()

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        workload["interval"],
        workload["actual_next_5min_tokens"],
        label="Actual workload",
        linewidth=0.9
    )

    alarm_intervals = alarm_test.loc[
        alarm_test["change_alarm"] == 1,
        "interval"
    ]

    alarm_workload = workload[
        workload["interval"].isin(
            alarm_intervals
        )
    ]

    if len(alarm_workload) > 0:

        plt.scatter(
            alarm_workload["interval"],
            alarm_workload["actual_next_5min_tokens"],
            label="Change-point alarm",
            marker="x",
            s=40
        )

    plt.xlabel(
        "Five-minute interval"
    )

    plt.ylabel(
        "Response tokens"
    )

    plt.title(
        "Paper 3 Change-Point Alarms on the Locked Test Period"
    )

    plt.legend()

    save_figure(
        "fig04_change_point_alarms.png"
    )

    # ==================================================================
    # FIGURE 5
    # FAILURE RISK DISTRIBUTION
    # ==================================================================

    print(
        "\n[5] Failure-risk probability distribution..."
    )

    failure_prob = (
        risk["calibrated_failure_probability"]
        .astype(float)
    )

    failure = (
        risk["failure"]
        .astype(int)
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.hist(
        failure_prob[failure == 0],
        bins=40,
        alpha=0.7,
        label="No failure"
    )

    plt.hist(
        failure_prob[failure == 1],
        bins=40,
        alpha=0.7,
        label="Failure"
    )

    plt.axvline(
        0.9591,
        linestyle="--",
        linewidth=1.2,
        label="Base threshold = 0.9591"
    )

    plt.xlabel(
        "Calibrated failure probability"
    )

    plt.ylabel(
        "Number of minutes"
    )

    plt.title(
        "Calibrated Failure-Risk Probability Distribution"
    )

    plt.legend()

    save_figure(
        "fig05_failure_risk_distribution.png"
    )

    # ==================================================================
    # FIGURE 6
    # CALIBRATION CURVE
    # ==================================================================

    print(
        "\n[6] Failure-risk calibration curve..."
    )

    probabilities = (
        risk["calibrated_failure_probability"]
        .astype(float)
        .to_numpy()
    )

    outcomes = (
        risk["failure"]
        .astype(int)
        .to_numpy()
    )

    bins = np.linspace(
        0,
        1,
        11
    )

    mean_predicted = []
    observed_rate = []
    counts = []

    for i in range(10):

        if i == 9:

            mask = (
                (probabilities >= bins[i])
                &
                (probabilities <= bins[i + 1])
            )

        else:

            mask = (
                (probabilities >= bins[i])
                &
                (probabilities < bins[i + 1])
            )

        if np.sum(mask) == 0:
            continue

        mean_predicted.append(
            np.mean(
                probabilities[mask]
            )
        )

        observed_rate.append(
            np.mean(
                outcomes[mask]
            )
        )

        counts.append(
            np.sum(mask)
        )

    plt.figure(
        figsize=(7, 7)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.scatter(
        mean_predicted,
        observed_rate,
        s=50,
        label="Calibrated model"
    )

    plt.xlabel(
        "Mean predicted probability"
    )

    plt.ylabel(
        "Observed failure frequency"
    )

    plt.title(
        "Failure-Risk Calibration — Locked Test Set"
    )

    plt.legend()

    save_figure(
        "fig06_failure_risk_calibration.png"
    )

    # ==================================================================
    # FIGURE 7
    # CAPACITY POLICY COMPARISON
    # ==================================================================

    print(
        "\n[7] Capacity policy comparison..."
    )

    policy_order = [
        "V1",
        "V2",
        "V3",
        "V4"
    ]

    capacity_plot = (
        capacity_metrics
        .set_index("policy")
        .loc[policy_order]
        .reset_index()
    )

    x = np.arange(
        len(capacity_plot)
    )

    width = 0.35

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        x - width / 2,
        capacity_plot[
            "under_provisioned_percent"
        ],
        width,
        label="Under-provisioned (%)"
    )

    plt.bar(
        x + width / 2,
        capacity_plot[
            "reserve_activation_percent"
        ],
        width,
        label="Reserve activation (%)"
    )

    plt.xticks(
        x,
        policy_order
    )

    plt.ylabel(
        "Percentage"
    )

    plt.xlabel(
        "Capacity policy"
    )

    plt.title(
        "Capacity Policy Outcomes on Locked Test Set"
    )

    plt.legend()

    save_figure(
        "fig07_capacity_policy_comparison.png"
    )

    # ==================================================================
    # FIGURE 8
    # SHORTAGE DISTRIBUTION
    # ==================================================================

    print(
        "\n[8] Shortage distributions..."
    )

    policy_data = {
        "V1": v1,
        "V2": v2,
        "V3": v3,
        "V4": v4
    }

    shortages = []

    for name in policy_order:

        df = policy_data[name]

        if name == "V1":

            capacity_column = (
                "V1_provisioned_capacity"
            )

        elif name == "V2":

            capacity_column = (
                "V2_provisioned_capacity"
            )

        elif name == "V3":

            capacity_column = (
                "V3_provisioned_capacity"
            )

        else:

            capacity_column = (
                "V4_provisioned_capacity"
            )

        shortage = np.maximum(
            df["actual_workload"].to_numpy()
            -
            df[capacity_column].to_numpy(),
            0
        )

        shortages.append(
            shortage
        )

    plt.figure(
        figsize=(9, 5)
    )

    plt.boxplot(
        shortages,
        labels=policy_order,
        showfliers=False
    )

    plt.xlabel(
        "Capacity policy"
    )

    plt.ylabel(
        "Shortage"
    )

    plt.title(
        "Capacity Shortage Distribution"
    )

    save_figure(
        "fig08_shortage_distribution.png"
    )

    # ==================================================================
    # FIGURE 9
    # RESERVE ACTIVATION
    # ==================================================================

    print(
        "\n[9] Reserve activation..."
    )

    activation = (
        capacity_plot[
            "reserve_activation_percent"
        ]
        .astype(float)
        .to_numpy()
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        policy_order,
        activation
    )

    plt.xlabel(
        "Capacity policy"
    )

    plt.ylabel(
        "Reserve activation (%)"
    )

    plt.title(
        "Reserve Activation Rate on Locked Test Set"
    )

    save_figure(
        "fig09_reserve_activation.png"
    )

    # ==================================================================
    # FIGURE 10
    # V3 VS V4 THRESHOLD / RESET
    # ==================================================================

    print(
        "\n[10] Adaptive threshold and change-point reset..."
    )

    threshold_df = v4[
        [
            "interval",
            "adaptive_threshold",
            "V4_threshold",
            "change_point_alarm"
        ]
    ].copy()

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        threshold_df["interval"],
        threshold_df["adaptive_threshold"],
        label="V3 adaptive threshold",
        linewidth=0.9
    )

    plt.plot(
        threshold_df["interval"],
        threshold_df["V4_threshold"],
        label="V4 threshold after reset",
        linewidth=0.9
    )

    reset_intervals = threshold_df.loc[
        threshold_df["change_point_alarm"] == 1,
        "interval"
    ]

    for interval in reset_intervals:

        plt.axvline(
            interval,
            linestyle=":",
            linewidth=0.8
        )

    plt.axhline(
        0.9591,
        linestyle="--",
        linewidth=1.0,
        label="Base threshold = 0.9591"
    )

    plt.xlabel(
        "Five-minute interval"
    )

    plt.ylabel(
        "Failure-risk threshold"
    )

    plt.title(
        "Adaptive Risk Threshold and Change-Point Resets"
    )

    plt.legend()

    save_figure(
        "fig10_threshold_reset_behavior.png"
    )

    # ==================================================================
    # FIGURE 11
    # BOOTSTRAP CONFIDENCE INTERVALS
    # ==================================================================

    print(
        "\n[11] Bootstrap comparison intervals..."
    )

    bootstrap_plot = bootstrap.copy()

    labels = []

    for _, row in bootstrap_plot.iterrows():

        labels.append(
            f"{row['comparison']}\n"
            f"{row['metric']}"
        )

    estimates = (
        bootstrap_plot[
            "observed_difference"
        ]
        .astype(float)
        .to_numpy()
    )

    lower = (
        bootstrap_plot[
            "ci_95_lower"
        ]
        .astype(float)
        .to_numpy()
    )

    upper = (
        bootstrap_plot[
            "ci_95_upper"
        ]
        .astype(float)
        .to_numpy()
    )

    y = np.arange(
        len(estimates)
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.errorbar(
        estimates,
        y,
        xerr=[
            estimates - lower,
            upper - estimates
        ],
        fmt="o",
        capsize=4
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1.0
    )

    plt.yticks(
        y,
        labels
    )

    plt.xlabel(
        "Observed paired difference with 95% CI"
    )

    plt.title(
        "Moving-Block Bootstrap Comparison Results"
    )

    save_figure(
        "fig11_bootstrap_confidence_intervals.png"
    )

    # ==================================================================
    # SAVE FIGURE MANIFEST
    # ==================================================================

    print(
        "\nSaving figure manifest..."
    )

    manifest = {

        "module":
            "14_make_figures",

        "figure_count":
            11,

        "figures": [

            "fig01_forecast_error_comparison.png",

            "fig02_actual_vs_forecast.png",

            "fig03_forecast_absolute_error.png",

            "fig04_change_point_alarms.png",

            "fig05_failure_risk_distribution.png",

            "fig06_failure_risk_calibration.png",

            "fig07_capacity_policy_comparison.png",

            "fig08_shortage_distribution.png",

            "fig09_reserve_activation.png",

            "fig10_threshold_reset_behavior.png",

            "fig11_bootstrap_confidence_intervals.png"
        ],

        "dpi":
            DPI,

        "test_data_used_for_model_training":
            False,

        "test_parameter_tuning":
            False
    }

    manifest_path = (
        FIGURES_DIR
        / "figure_manifest.json"
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
        "  results/figures/figure_manifest.json"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODULE 14 COMPLETE"
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
        "Figures were generated from existing results only."
    )


if __name__ == "__main__":
    main()