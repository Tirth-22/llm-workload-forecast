"""
02_build_time_views.py

Build the two time-resolution views required by the research:

1. Minute-level view
   - Used later for Paper 4-style failure-risk modeling.

2. Five-minute view
   - Used later for Paper 3 change-point detection
     and short-term workload forecasting.

Important:
- Raw duplicate records are retained.
- Zero-response records are retained for the risk view.
- Empty time intervals are explicitly represented with zeros.
- Timestamp is an anonymized relative-second clock.
- No real date/timezone is inferred.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RAW_FILE = Path("data/BurstGPT_1.csv")

MINUTE_OUTPUT = Path(
    "data/burstgpt_minute.csv"
)

FIVE_MINUTE_OUTPUT = Path(
    "data/burstgpt_5min.csv"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "Timestamp",
    "Model",
    "Request tokens",
    "Response tokens",
    "Total tokens",
    "Log Type",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_raw_data():

    if not RAW_FILE.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{RAW_FILE.resolve()}\n\n"
            f"Make sure BurstGPT_1.csv is located at:\n"
            f"data/BurstGPT_1.csv"
        )

    print("\nLoading raw BurstGPT dataset...")

    df = pd.read_csv(RAW_FILE)

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(
        f"Raw rows loaded: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# AGGREGATION FUNCTION
# ============================================================

def build_time_view(df, interval_seconds):

    print(
        f"\nBuilding "
        f"{interval_seconds}-second view..."
    )

    data = df.copy()

    # --------------------------------------------------------
    # Assign interval index
    # --------------------------------------------------------

    data["interval"] = (
        data["Timestamp"]
        // interval_seconds
    ).astype("int64")

    # --------------------------------------------------------
    # Model/service indicators
    # --------------------------------------------------------

    data["is_gpt4"] = (
        data["Model"].astype(str)
        == "GPT-4"
    ).astype("int64")

    data["is_api"] = (
        data["Log Type"].astype(str)
        == "API log"
    ).astype("int64")

    data["is_zero_response"] = (
        data["Response tokens"] == 0
    ).astype("int64")

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    grouped = (
        data
        .groupby(
            "interval",
            sort=True
        )
        .agg(
            request_count=(
                "Timestamp",
                "size"
            ),

            request_tokens=(
                "Request tokens",
                "sum"
            ),

            response_tokens=(
                "Response tokens",
                "sum"
            ),

            total_tokens=(
                "Total tokens",
                "sum"
            ),

            zero_response_count=(
                "is_zero_response",
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
        .reset_index()
    )

    # --------------------------------------------------------
    # Complete interval grid
    # --------------------------------------------------------

    first_interval = int(
        data["interval"].min()
    )

    last_interval = int(
        data["interval"].max()
    )

    complete_grid = pd.DataFrame(
        {
            "interval": np.arange(
                first_interval,
                last_interval + 1,
                dtype=np.int64,
            )
        }
    )

    result = (
        complete_grid
        .merge(
            grouped,
            on="interval",
            how="left"
        )
    )

    # --------------------------------------------------------
    # Fill empty intervals
    # --------------------------------------------------------

    count_columns = [
        "request_count",
        "request_tokens",
        "response_tokens",
        "total_tokens",
        "zero_response_count",
        "gpt4_count",
        "api_count",
    ]

    result[count_columns] = (
        result[count_columns]
        .fillna(0)
        .astype("int64")
    )

    # --------------------------------------------------------
    # Shares
    # --------------------------------------------------------

    result["gpt4_share"] = np.where(
        result["request_count"] > 0,

        result["gpt4_count"]
        / result["request_count"],

        0.0,
    )

    result["api_share"] = np.where(
        result["request_count"] > 0,

        result["api_count"]
        / result["request_count"],

        0.0,
    )

    # --------------------------------------------------------
    # Relative time boundaries
    # --------------------------------------------------------

    result["start_second"] = (
        result["interval"]
        * interval_seconds
    )

    result["end_second_exclusive"] = (
        result["start_second"]
        + interval_seconds
    )

    # --------------------------------------------------------
    # Column ordering
    # --------------------------------------------------------

    result = result[
        [
            "interval",
            "start_second",
            "end_second_exclusive",

            "request_count",
            "request_tokens",
            "response_tokens",
            "total_tokens",

            "zero_response_count",

            "gpt4_count",
            "gpt4_share",

            "api_count",
            "api_share",
        ]
    ]

    return result


# ============================================================
# REMOVE FINAL INCOMPLETE FIVE-MINUTE BIN
# ============================================================

def remove_incomplete_final_bin(
    five_minute_df,
    raw_df
):

    max_timestamp = int(
        raw_df["Timestamp"].max()
    )

    final_interval = (
        max_timestamp // 300
    )

    final_interval_end = (
        final_interval + 1
    ) * 300

    # If the final timestamp does not reach
    # the end of its five-minute interval,
    # that interval is incomplete.

    if max_timestamp < (
        final_interval_end - 1
    ):

        before = len(
            five_minute_df
        )

        five_minute_df = (
            five_minute_df[
                five_minute_df["interval"]
                != final_interval
            ]
            .copy()
        )

        after = len(
            five_minute_df
        )

        print(
            "\nRemoved incomplete final "
            "five-minute interval:"
        )

        print(
            f"  Interval: {final_interval}"
        )

        print(
            f"  Rows before: {before:,}"
        )

        print(
            f"  Rows after : {after:,}"
        )

    else:

        print(
            "\nFinal five-minute interval "
            "is complete."
        )

    return five_minute_df


# ============================================================
# VALIDATE OUTPUT
# ============================================================

def validate_time_view(
    df,
    interval_seconds
):

    print(
        f"\nValidating "
        f"{interval_seconds}-second view..."
    )

    # --------------------------------------------------------
    # Interval continuity
    # --------------------------------------------------------

    differences = (
        df["interval"]
        .diff()
        .dropna()
    )

    missing_intervals = int(
        (differences != 1).sum()
    )

    print(
        f"Missing interval gaps: "
        f"{missing_intervals:,}"
    )

    # --------------------------------------------------------
    # Negative values
    # --------------------------------------------------------

    numeric_columns = [
        "request_count",
        "request_tokens",
        "response_tokens",
        "total_tokens",
        "zero_response_count",
        "gpt4_count",
        "api_count",
    ]

    negative_values = 0

    for column in numeric_columns:

        negative_values += int(
            (df[column] < 0).sum()
        )

    print(
        f"Negative aggregate values: "
        f"{negative_values:,}"
    )

    # --------------------------------------------------------
    # Arithmetic
    # --------------------------------------------------------

    arithmetic_errors = int(
        (
            df["total_tokens"]
            !=
            df["request_tokens"]
            + df["response_tokens"]
        ).sum()
    )

    print(
        f"Token arithmetic errors: "
        f"{arithmetic_errors:,}"
    )

    # --------------------------------------------------------
    # Zero response sanity
    # --------------------------------------------------------

    invalid_zero_response = int(
        (
            df["zero_response_count"]
            > df["request_count"]
        ).sum()
    )

    print(
        "Invalid zero-response counts: "
        f"{invalid_zero_response:,}"
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if (
        missing_intervals == 0
        and negative_values == 0
        and arithmetic_errors == 0
        and invalid_zero_response == 0
    ):

        print(
            "View validation: PASS"
        )

    else:

        raise ValueError(
            "Time-view validation failed."
        )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    name,
    df
):

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(
        f"Rows: "
        f"{len(df):,}"
    )

    print(
        f"Interval range: "
        f"{df['interval'].min():,}"
        f" -> "
        f"{df['interval'].max():,}"
    )

    print(
        f"Request count: "
        f"{df['request_count'].sum():,}"
    )

    print(
        f"Request tokens: "
        f"{df['request_tokens'].sum():,}"
    )

    print(
        f"Response tokens: "
        f"{df['response_tokens'].sum():,}"
    )

    print(
        f"Total tokens: "
        f"{df['total_tokens'].sum():,}"
    )

    print(
        f"Zero-response records: "
        f"{df['zero_response_count'].sum():,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BURSTGPT TIME-VIEW CONSTRUCTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    raw_df = load_raw_data()

    # --------------------------------------------------------
    # Minute view
    # --------------------------------------------------------

    minute_df = build_time_view(
        raw_df,
        interval_seconds=60
    )

    # --------------------------------------------------------
    # Five-minute view
    # --------------------------------------------------------

    five_minute_df = build_time_view(
        raw_df,
        interval_seconds=300
    )

    # --------------------------------------------------------
    # Remove incomplete final five-minute interval
    # --------------------------------------------------------

    five_minute_df = (
        remove_incomplete_final_bin(
            five_minute_df,
            raw_df
        )
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_time_view(
        minute_df,
        interval_seconds=60
    )

    validate_time_view(
        five_minute_df,
        interval_seconds=300
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    MINUTE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    FIVE_MINUTE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    minute_df.to_csv(
        MINUTE_OUTPUT,
        index=False
    )

    five_minute_df.to_csv(
        FIVE_MINUTE_OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # Print summaries
    # --------------------------------------------------------

    print_summary(
        "MINUTE VIEW",
        minute_df
    )

    print_summary(
        "FIVE-MINUTE VIEW",
        five_minute_df
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TIME-VIEW CONSTRUCTION COMPLETE")
    print("=" * 70)

    print(
        "\nFiles created:"
    )

    print(
        f"  {MINUTE_OUTPUT}"
    )

    print(
        f"  {FIVE_MINUTE_OUTPUT}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()