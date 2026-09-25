"""
04_build_workload_features.py

Builds the causal feature matrix for Paper 3 workload forecasting.

Target:
    next five-minute total-token workload

Features:
    Lags:
        1, 2, 3, 6, 12, 24, 72, 144, 288

    Shifted rolling mean/std:
        3, 12, 72, 288

    Shifted rolling max:
        12, 72, 288

    Calendar features:
        daily sin/cos
        weekly sin/cos

CP-HGB-only features:
    change_score
    change_alarm
    short_long_ratio
    elapsed_bins_since_alarm

IMPORTANT:
All workload/history features are shifted by one interval.
The future target is never used as an input feature.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "results/alarms/change_point_5min.csv"
)

OUTPUT_FILE = Path(
    "data/workload_features_5min.csv"
)

TARGET_COLUMN = "total_tokens"

LAGS = [
    1,
    2,
    3,
    6,
    12,
    24,
    72,
    144,
    288,
]

ROLLING_MEAN_STD = [
    3,
    12,
    72,
    288,
]

ROLLING_MAX = [
    12,
    72,
    288,
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n[1] Loading change-point output...")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nInput file not found:\n"
            f"{INPUT_FILE.resolve()}\n\n"
            f"Run 03_change_point_detector.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "interval",
        "total_tokens",
        "change_score",
        "change_alarm",
        "short_long_ratio",
        "elapsed_bins_since_alarm",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(
        f"Rows loaded: {len(df):,}"
    )

    return df


# ============================================================
# BUILD FEATURES
# ============================================================

def build_features(df):

    print(
        "\n[2] Building causal workload features..."
    )

    result = df.copy()

    y = result[
        TARGET_COLUMN
    ].astype(float)

    # --------------------------------------------------------
    # LAG FEATURES
    # --------------------------------------------------------

    print("\nCreating lag features...")

    for lag in LAGS:

        column_name = (
            f"lag_{lag}"
        )

        result[column_name] = (
            y.shift(lag)
        )

        print(
            f"  {column_name}"
        )

    # --------------------------------------------------------
    # SHIFTED ROLLING MEAN + STD
    # --------------------------------------------------------

    print(
        "\nCreating shifted rolling "
        "mean/std features..."
    )

    history = y.shift(1)

    for window in ROLLING_MEAN_STD:

        mean_name = (
            f"rolling_mean_{window}"
        )

        std_name = (
            f"rolling_std_{window}"
        )

        result[mean_name] = (
            history
            .rolling(
                window=window,
                min_periods=window
            )
            .mean()
        )

        result[std_name] = (
            history
            .rolling(
                window=window,
                min_periods=window
            )
            .std()
        )

        print(
            f"  {mean_name}"
        )

        print(
            f"  {std_name}"
        )

    # --------------------------------------------------------
    # SHIFTED ROLLING MAX
    # --------------------------------------------------------

    print(
        "\nCreating shifted rolling "
        "maximum features..."
    )

    for window in ROLLING_MAX:

        column_name = (
            f"rolling_max_{window}"
        )

        result[column_name] = (
            history
            .rolling(
                window=window,
                min_periods=window
            )
            .max()
        )

        print(
            f"  {column_name}"
        )

    # --------------------------------------------------------
    # CALENDAR FEATURES
    # --------------------------------------------------------

    print(
        "\nCreating calendar features..."
    )

    # Five-minute interval index.
    #
    # 12 bins = 1 hour
    # 288 bins = 1 day
    # 2016 bins = 1 week

    interval = (
        result["interval"]
        .astype(float)
    )

    # Position within day:
    # 0 ... 287

    position_in_day = (
        interval % 288
    )

    daily_angle = (
        2
        * np.pi
        * position_in_day
        / 288
    )

    result["daily_sin"] = (
        np.sin(daily_angle)
    )

    result["daily_cos"] = (
        np.cos(daily_angle)
    )

    # Position within week:
    # 0 ... 2015

    position_in_week = (
        interval % 2016
    )

    weekly_angle = (
        2
        * np.pi
        * position_in_week
        / 2016
    )

    result["weekly_sin"] = (
        np.sin(weekly_angle)
    )

    result["weekly_cos"] = (
        np.cos(weekly_angle)
    )

    print(
        "  daily_sin"
    )

    print(
        "  daily_cos"
    )

    print(
        "  weekly_sin"
    )

    print(
        "  weekly_cos"
    )

    # --------------------------------------------------------
    # FUTURE TARGET
    # --------------------------------------------------------

    print(
        "\nCreating next-interval target..."
    )

    result[
        "target_next_5min"
    ] = y.shift(-1)

    # --------------------------------------------------------
    # CP-HGB FEATURES
    # --------------------------------------------------------

    print(
        "\nAdding change-point features..."
    )

    # These values describe the state available
    # at the current forecast origin.

    result[
        "cp_change_score"
    ] = result[
        "change_score"
    ]

    result[
        "cp_change_alarm"
    ] = result[
        "change_alarm"
    ]

    result[
        "cp_short_long_ratio"
    ] = result[
        "short_long_ratio"
    ]

    result[
        "cp_elapsed_bins"
    ] = result[
        "elapsed_bins_since_alarm"
    ]

    # --------------------------------------------------------
    # Mark rows with sufficient history
    # --------------------------------------------------------

    minimum_history = max(
        LAGS
    )

    result[
        "has_full_history"
    ] = (
        result.index
        >= minimum_history
    )

    # --------------------------------------------------------
    # Select useful columns
    # --------------------------------------------------------

    feature_columns = [
        "interval",

        TARGET_COLUMN,

        "target_next_5min",

        # HGB features
        *[
            f"lag_{lag}"
            for lag in LAGS
        ],

        *[
            f"rolling_mean_{window}"
            for window in ROLLING_MEAN_STD
        ],

        *[
            f"rolling_std_{window}"
            for window in ROLLING_MEAN_STD
        ],

        *[
            f"rolling_max_{window}"
            for window in ROLLING_MAX
        ],

        "daily_sin",
        "daily_cos",
        "weekly_sin",
        "weekly_cos",

        # CP-HGB features
        "cp_change_score",
        "cp_change_alarm",
        "cp_short_long_ratio",
        "cp_elapsed_bins",

        "has_full_history",
    ]

    result = result[
        feature_columns
    ]

    return result


# ============================================================
# VALIDATE FEATURES
# ============================================================

def validate_features(
    features
):

    print(
        "\n[3] Validating feature matrix..."
    )

    # --------------------------------------------------------
    # Expected feature counts
    # --------------------------------------------------------

    hgb_feature_count = (
        len(LAGS)
        + len(ROLLING_MEAN_STD) * 2
        + len(ROLLING_MAX)
        + 4
    )

    cp_feature_count = 4

    expected_total = (
        1                       # interval
        + 1                     # current target
        + 1                     # future target
        + hgb_feature_count
        + cp_feature_count
        + 1                     # history flag
    )

    actual_total = len(
        features.columns
    )

    print(
        f"Expected columns: "
        f"{expected_total}"
    )

    print(
        f"Actual columns:   "
        f"{actual_total}"
    )

    if actual_total != expected_total:

        raise ValueError(
            "Unexpected feature-column count."
        )

    # --------------------------------------------------------
    # Check future target
    # --------------------------------------------------------

    target_missing = int(
        features[
            "target_next_5min"
        ]
        .isna()
        .sum()
    )

    print(
        f"Rows without future target: "
        f"{target_missing:,}"
    )

    # --------------------------------------------------------
    # Check feature NaNs
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in features.columns
        if column not in [
            "interval",
            TARGET_COLUMN,
            "target_next_5min",
            "has_full_history",
        ]
    ]

    print(
        "\nNaN counts in model features:"
    )

    total_feature_nans = 0

    for column in feature_columns:

        count = int(
            features[column]
            .isna()
            .sum()
        )

        total_feature_nans += count

        if count > 0:

            print(
                f"  {column}: "
                f"{count:,}"
            )

    print(
        f"\nTotal feature NaNs: "
        f"{total_feature_nans:,}"
    )

    # --------------------------------------------------------
    # Verify causal boundary
    # --------------------------------------------------------

    first_full_history = (
        features[
            features["has_full_history"]
        ]
        .index
        .min()
    )

    print(
        f"First full-history row: "
        f"{first_full_history}"
    )

    if first_full_history != max(LAGS):

        raise ValueError(
            "Unexpected full-history boundary."
        )

    # --------------------------------------------------------
    # Verify target shift
    # --------------------------------------------------------

    original = features[
        TARGET_COLUMN
    ].to_numpy()

    shifted = features[
        "target_next_5min"
    ].to_numpy()

    valid_positions = (
        ~np.isnan(shifted)
    )

    expected_target = (
        original[1:]
    )

    actual_target = (
        shifted[:-1]
    )

    if not np.allclose(
        actual_target[
            valid_positions[:-1]
        ],
        expected_target[
            valid_positions[:-1]
        ],
    ):

        raise ValueError(
            "Future target shift is incorrect."
        )

    print(
        "Future-target alignment: PASS"
    )

    print(
        "Feature validation: PASS"
    )


# ============================================================
# SAVE
# ============================================================

def save_features(
    features
):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\n[4] Feature matrix saved:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    features
):

    print(
        "\n" + "-" * 70
    )

    print(
        "WORKLOAD FEATURE SUMMARY"
    )

    print(
        "-" * 70
    )

    print(
        f"Rows: "
        f"{len(features):,}"
    )

    print(
        f"Columns: "
        f"{len(features.columns):,}"
    )

    print(
        f"HGB features: "
        f"{len(LAGS) + len(ROLLING_MEAN_STD) * 2 + len(ROLLING_MAX) + 4}"
    )

    print(
        f"CP-HGB additional features: "
        f"{4}"
    )

    print(
        f"Target: "
        f"target_next_5min"
    )

    print(
        f"Maximum lag: "
        f"{max(LAGS)} bins"
    )

    print(
        f"Maximum history: "
        f"{max(LAGS) * 5} minutes"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "PAPER 3 WORKLOAD FEATURE ENGINEERING"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    features = build_features(
        df
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_features(
        features
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_features(
        features
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        features
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "WORKLOAD FEATURE ENGINEERING COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()