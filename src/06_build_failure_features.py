"""
06_build_failure_features.py

Paper 4 - Minute-level zero-response failure-risk features.

Dataset:
    BurstGPT_1.csv

Timestamp:
    Integer elapsed seconds from an anonymized trace origin.
    It is NOT a Unix timestamp and has no civil date/timezone.

Target:
    failure = 1 if at least one raw request in a minute
              has Response tokens == 0
              else 0.

89 risk predictors:

    Five source series:
        1. request_count
        2. completed_tokens
        3. gpt4_share
        4. api_share
        5. zero_response_count

    For each source:
        7 lags
        5 rolling sums
        5 rolling means

    17 × 5 = 85 temporal predictors

    Calendar:
        minute-of-day sin/cos
        trace-day-of-week sin/cos

    85 + 4 = 89 predictors

Causality:
    All lag and rolling features use information from
    prior minutes only.

Important:
    Zero-response is treated as a trace-observed
    noncompletion signal. It is NOT interpreted as
    definite GPU overload or resource exhaustion.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RAW_FILE = Path(
    "data/BurstGPT_1.csv"
)

OUTPUT_FILE = Path(
    "data/failure_features_minute.csv"
)

# Paper 4 risk feature specification
LAGS = [
    1,
    2,
    5,
    15,
    60,
    240,
    1440,
]

ROLLING_WINDOWS = [
    5,
    15,
    60,
    240,
    1440,
]


# ============================================================
# COLUMN HELPER
# ============================================================

def find_column(df, candidates):

    normalized = {
        str(column).strip().lower():
            column
        for column in df.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in normalized:

            return normalized[key]

    raise ValueError(
        f"Could not find any of these columns: "
        f"{candidates}\n"
        f"Available columns: "
        f"{list(df.columns)}"
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_raw_data():

    print(
        "\n[1] Loading BurstGPT raw data..."
    )

    if not RAW_FILE.exists():

        raise FileNotFoundError(
            f"Raw file not found:\n"
            f"{RAW_FILE.resolve()}"
        )

    df = pd.read_csv(
        RAW_FILE
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    return df


# ============================================================
# IDENTIFY SCHEMA
# ============================================================

def prepare_schema(df):

    print(
        "\n[2] Identifying BurstGPT columns..."
    )

    timestamp_col = find_column(
        df,
        [
            "Timestamp",
            "timestamp",
        ]
    )

    model_col = find_column(
        df,
        [
            "Model",
            "model",
        ]
    )

    request_tokens_col = find_column(
        df,
        [
            "Request tokens",
            "Request Tokens",
            "request_tokens",
        ]
    )

    response_tokens_col = find_column(
        df,
        [
            "Response tokens",
            "Response Tokens",
            "response_tokens",
        ]
    )

    print(
        f"Timestamp column       : "
        f"{timestamp_col}"
    )

    print(
        f"Model column           : "
        f"{model_col}"
    )

    print(
        f"Request tokens column  : "
        f"{request_tokens_col}"
    )

    print(
        f"Response tokens column : "
        f"{response_tokens_col}"
    )

    return (
        timestamp_col,
        model_col,
        request_tokens_col,
        response_tokens_col,
    )


# ============================================================
# BUILD MINUTE AGGREGATES
# ============================================================

def build_minute_aggregates(
    df,
    timestamp_col,
    model_col,
    request_tokens_col,
    response_tokens_col,
):

    print(
        "\n[3] Building minute-level aggregates..."
    )

    work = df[
        [
            timestamp_col,
            model_col,
            request_tokens_col,
            response_tokens_col,
        ]
    ].copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------
    #
    # Paper 4 describes BurstGPT Timestamp as:
    #
    #   relative seconds
    #
    # The values in the user's file are:
    #
    #   5, 45, 118, ...
    #
    # Therefore they must NOT be passed to
    # pd.to_datetime().
    #
    # Minute index:
    #
    #   floor(relative_seconds / 60)
    #
    # --------------------------------------------------------

    work[timestamp_col] = pd.to_numeric(
        work[timestamp_col],
        errors="coerce"
    )

    invalid_timestamp = (
        work[timestamp_col].isna().sum()
    )

    if invalid_timestamp > 0:

        raise ValueError(
            f"Invalid timestamps found: "
            f"{invalid_timestamp:,}"
        )

    if (
        work[timestamp_col] < 0
    ).any():

        raise ValueError(
            "Negative relative timestamps found."
        )

    # Check chronological ordering.
    if not work[timestamp_col].is_monotonic_increasing:

        raise ValueError(
            "BurstGPT timestamps are not "
            "nondecreasing."
        )

    # --------------------------------------------------------
    # Token columns
    # --------------------------------------------------------

    work[request_tokens_col] = pd.to_numeric(
        work[request_tokens_col],
        errors="coerce"
    )

    work[response_tokens_col] = pd.to_numeric(
        work[response_tokens_col],
        errors="coerce"
    )

    invalid_tokens = (
        work[
            [
                request_tokens_col,
                response_tokens_col,
            ]
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    if invalid_tokens > 0:

        raise ValueError(
            f"Invalid token values found: "
            f"{invalid_tokens:,}"
        )

    # --------------------------------------------------------
    # Relative minute index
    # --------------------------------------------------------

    work["minute"] = (
        work[timestamp_col]
        // 60
    ).astype(int)

    # --------------------------------------------------------
    # Zero-response indicator
    # --------------------------------------------------------

    work["zero_response"] = (
        work[response_tokens_col] == 0
    ).astype(int)

    # --------------------------------------------------------
    # Model indicator
    # --------------------------------------------------------

    model_text = (
        work[model_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    work["is_gpt4"] = (
        model_text
        .str.contains(
            "gpt-4",
            regex=False,
            na=False
        )
    ).astype(int)

    # --------------------------------------------------------
    # Service indicator
    # --------------------------------------------------------
    #
    # Paper 4 defines API share using the service type.
    # The raw field is "Log Type", but this function only
    # receives Model/token columns.
    #
    # Therefore we need to identify service information
    # separately before aggregation.
    #
    # We handle it below using the original dataframe.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Identify Log Type from original dataframe
    # --------------------------------------------------------

    # Find service column.
    service_col = find_column(
        df,
        [
            "Log Type",
            "log type",
            "LogType",
            "service",
        ]
    )

    service_text = (
        df[service_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    work["is_api"] = (
        service_text
        .str.contains(
            "api",
            regex=False,
            na=False
        )
    ).astype(int)

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    grouped = (
        work
        .groupby("minute")
        .agg(
            request_count=(
                timestamp_col,
                "size"
            ),

            completed_tokens=(
                response_tokens_col,
                lambda x:
                    x[x > 0].sum()
            ),

            zero_response_count=(
                "zero_response",
                "sum"
            ),

            gpt4_count=(
                "is_gpt4",
                "sum"
            ),

            api_count=(
                "is_api",
                "sum"
            ),
        )
        .sort_index()
    )

    # --------------------------------------------------------
    # Shares
    # --------------------------------------------------------

    grouped["gpt4_share"] = (
        grouped["gpt4_count"]
        / grouped["request_count"]
    )

    grouped["api_share"] = (
        grouped["api_count"]
        / grouped["request_count"]
    )

    # --------------------------------------------------------
    # Failure target
    # --------------------------------------------------------

    grouped["failure"] = (
        grouped["zero_response_count"] > 0
    ).astype(int)

    # --------------------------------------------------------
    # IMPORTANT:
    # Fill absent minute indices with zero.
    #
    # Paper 4 states that minute aggregates are represented
    # on a complete minute-index grid with absent intervals
    # filled by zero.
    #
    # The relative clock starts at minute 0.
    # --------------------------------------------------------

    min_minute = int(
        grouped.index.min()
    )

    max_minute = int(
        grouped.index.max()
    )

    complete_index = pd.RangeIndex(
        start=min_minute,
        stop=max_minute + 1,
        step=1,
        name="minute",
    )

    grouped = grouped.reindex(
        complete_index,
        fill_value=0
    )

    # --------------------------------------------------------
    # Recalculate failure after zero filling
    # --------------------------------------------------------

    grouped["failure"] = (
        grouped["zero_response_count"] > 0
    ).astype(int)

    # --------------------------------------------------------
    # Keep required source series + target
    # --------------------------------------------------------

    minute = grouped[
        [
            "request_count",
            "completed_tokens",
            "gpt4_share",
            "api_share",
            "zero_response_count",
            "failure",
        ]
    ].copy()

    print(
        f"Observed minute indices: "
        f"{len(minute):,}"
    )

    print(
        f"Minute index range: "
        f"{minute.index.min()} "
        f"-> "
        f"{minute.index.max()}"
    )

    print(
        f"Failure minutes: "
        f"{minute['failure'].sum():,}"
    )

    print(
        f"Failure prevalence: "
        f"{minute['failure'].mean() * 100:.4f}%"
    )

    return minute


# ============================================================
# ADD TEMPORAL FEATURES
# ============================================================

def add_temporal_features(
    minute
):

    print(
        "\n[4] Creating causal temporal features..."
    )

    source_series = [
        "request_count",
        "completed_tokens",
        "gpt4_share",
        "api_share",
        "zero_response_count",
    ]

    # --------------------------------------------------------
    # LAGS
    # --------------------------------------------------------

    print(
        "\nCreating lag features..."
    )

    for source in source_series:

        for lag in LAGS:

            column = (
                f"{source}_lag_{lag}"
            )

            minute[column] = (
                minute[source]
                .shift(lag)
            )

            print(
                f"  {column}"
            )

    # --------------------------------------------------------
    # ROLLING SUMS
    # --------------------------------------------------------

    print(
        "\nCreating rolling sum features..."
    )

    for source in source_series:

        for window in ROLLING_WINDOWS:

            column = (
                f"{source}_rolling_sum_{window}"
            )

            # Shift first.
            #
            # At minute t:
            #   uses t-1, t-2, ..., t-window
            #
            # It does NOT use the current minute.

            minute[column] = (
                minute[source]
                .shift(1)
                .rolling(
                    window=window,
                    min_periods=window,
                )
                .sum()
            )

            print(
                f"  {column}"
            )

    # --------------------------------------------------------
    # ROLLING MEANS
    # --------------------------------------------------------

    print(
        "\nCreating rolling mean features..."
    )

    for source in source_series:

        for window in ROLLING_WINDOWS:

            column = (
                f"{source}_rolling_mean_{window}"
            )

            minute[column] = (
                minute[source]
                .shift(1)
                .rolling(
                    window=window,
                    min_periods=window,
                )
                .mean()
            )

            print(
                f"  {column}"
            )

    # --------------------------------------------------------
    # TRACE-RELATIVE CALENDAR FEATURES
    # --------------------------------------------------------
    #
    # BurstGPT has an anonymized relative clock.
    #
    # Therefore:
    #
    #   minute-of-day:
    #       elapsed minute modulo 1440
    #
    #   trace-day:
    #       elapsed minute // 1440
    #
    #   day-of-week:
    #       trace-day modulo 7
    #
    # This does NOT claim a real-world weekday.
    # It is a deterministic relative calendar representation.
    # --------------------------------------------------------

    print(
        "\nCreating trace-relative calendar features..."
    )

    minute_index = (
        minute.index.to_numpy(
            dtype=int
        )
    )

    minute_of_day = (
        minute_index % 1440
    )

    trace_day = (
        minute_index // 1440
    )

    relative_day_of_week = (
        trace_day % 7
    )

    # --------------------------------------------------------
    # Minute of day
    # --------------------------------------------------------

    minute["minute_of_day_sin"] = (
        np.sin(
            2
            * np.pi
            * minute_of_day
            / 1440.0
        )
    )

    minute["minute_of_day_cos"] = (
        np.cos(
            2
            * np.pi
            * minute_of_day
            / 1440.0
        )
    )

    # --------------------------------------------------------
    # Relative trace day of week
    # --------------------------------------------------------

    minute["day_of_week_sin"] = (
        np.sin(
            2
            * np.pi
            * relative_day_of_week
            / 7.0
        )
    )

    minute["day_of_week_cos"] = (
        np.cos(
            2
            * np.pi
            * relative_day_of_week
            / 7.0
        )
    )

    print(
        "  minute_of_day_sin"
    )

    print(
        "  minute_of_day_cos"
    )

    print(
        "  day_of_week_sin"
    )

    print(
        "  day_of_week_cos"
    )

    return minute


# ============================================================
# VALIDATION
# ============================================================

def validate_features(
    minute
):

    print(
        "\n[5] Validating risk feature matrix..."
    )

    source_series = [
        "request_count",
        "completed_tokens",
        "gpt4_share",
        "api_share",
        "zero_response_count",
    ]

    temporal_features = []

    # --------------------------------------------------------
    # Build expected temporal feature names
    # --------------------------------------------------------

    for source in source_series:

        for lag in LAGS:

            temporal_features.append(
                f"{source}_lag_{lag}"
            )

        for window in ROLLING_WINDOWS:

            temporal_features.append(
                f"{source}_rolling_sum_{window}"
            )

        for window in ROLLING_WINDOWS:

            temporal_features.append(
                f"{source}_rolling_mean_{window}"
            )

    calendar_features = [
        "minute_of_day_sin",
        "minute_of_day_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    ]

    expected_features = (
        temporal_features
        + calendar_features
    )

    # --------------------------------------------------------
    # Feature count
    # --------------------------------------------------------

    actual_feature_count = len(
        [
            column
            for column in expected_features
            if column in minute.columns
        ]
    )

    print(
        f"Expected risk features: "
        f"{len(expected_features)}"
    )

    print(
        f"Actual risk features:   "
        f"{actual_feature_count}"
    )

    missing = [
        column
        for column in expected_features
        if column not in minute.columns
    ]

    if missing:

        raise ValueError(
            f"Missing features:\n"
            f"{missing}"
        )

    if (
        len(expected_features)
        != 89
    ):

        raise AssertionError(
            "Expected exactly 89 risk features."
        )

    # --------------------------------------------------------
    # Historical NaN validation
    # --------------------------------------------------------

    max_history = max(
        max(LAGS),
        max(ROLLING_WINDOWS),
    )

    print(
        "\nChecking initial historical NaNs..."
    )

    first_rows = minute.iloc[
        :max_history
    ]

    temporal_nan_count = (
        first_rows[
            temporal_features
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Temporal NaNs in initial "
        f"{max_history} rows: "
        f"{temporal_nan_count:,}"
    )

    # --------------------------------------------------------
    # No NaNs after history becomes available
    # --------------------------------------------------------

    usable = minute.iloc[
        max_history:
    ]

    remaining_nan = (
        usable[
            expected_features
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Feature NaNs after "
        f"{max_history} minutes: "
        f"{remaining_nan:,}"
    )

    if remaining_nan != 0:

        raise ValueError(
            "Unexpected NaNs remain after "
            "the required historical window."
        )

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    unique_target = sorted(
        minute["failure"]
        .unique()
        .tolist()
    )

    print(
        f"Failure target values: "
        f"{unique_target}"
    )

    if unique_target != [
        0,
        1,
    ]:

        raise ValueError(
            "Failure target must contain "
            "both 0 and 1."
        )

    # --------------------------------------------------------
    # Causal spot check
    # --------------------------------------------------------
    #
    # Verify lag-1 at minute t equals source at t-1.
    # Verify rolling-sum-5 at t equals sum of t-1...t-5.
    # --------------------------------------------------------

    print(
        "\nRunning causal spot checks..."
    )

    if len(minute) > 10:

        test_position = 10

        current_minute = (
            minute.index[
                test_position
            ]
        )

        previous_minute = (
            minute.index[
                test_position - 1
            ]
        )

        expected_lag = minute.loc[
            previous_minute,
            "request_count"
        ]

        actual_lag = minute.loc[
            current_minute,
            "request_count_lag_1"
        ]

        if not np.isclose(
            expected_lag,
            actual_lag
        ):

            raise AssertionError(
                "Lag-1 causality check failed."
            )

        previous_values = (
            minute[
                "request_count"
            ]
            .iloc[
                test_position - 5:
                test_position
            ]
        )

        expected_sum = (
            previous_values.sum()
        )

        actual_sum = minute.loc[
            current_minute,
            "request_count_rolling_sum_5"
        ]

        if not np.isclose(
            expected_sum,
            actual_sum
        ):

            raise AssertionError(
                "Rolling-sum causality check failed."
            )

    print(
        "Causal feature checks: PASS"
    )

    print(
        "\nFeature validation: PASS"
    )


# ============================================================
# SAVE
# ============================================================

def save_features(
    minute
):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        minute
        .reset_index()
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\n[6] Failure-risk feature matrix saved:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "PAPER 4 FAILURE-RISK FEATURE ENGINEERING"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_raw_data()

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    (
        timestamp_col,
        model_col,
        request_tokens_col,
        response_tokens_col,
    ) = prepare_schema(
        df
    )

    # --------------------------------------------------------
    # Minute aggregation
    # --------------------------------------------------------

    minute = build_minute_aggregates(
        df,
        timestamp_col,
        model_col,
        request_tokens_col,
        response_tokens_col,
    )

    # --------------------------------------------------------
    # Temporal features
    # --------------------------------------------------------

    minute = add_temporal_features(
        minute
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_features(
        minute
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_features(
        minute
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FAILURE-RISK FEATURE ENGINEERING COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()