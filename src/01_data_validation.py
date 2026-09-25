"""
01_data_validation.py

Purpose:
    Validate the original BurstGPT_1.csv before any preprocessing
    or machine-learning work.

Dataset expected from the BurstGPT trace:
    Timestamp
    Model
    Request tokens
    Response tokens
    Total tokens
    Log Type

Important:
    - Do NOT remove duplicate rows.
    - The dataset has no request ID, so identical rows may represent
      separate requests.
    - Do NOT modify the raw dataset in this step.
"""

from pathlib import Path
import hashlib
import json

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/BurstGPT_1.csv")

OUTPUT_DIR = Path("results/validation")
OUTPUT_FILE = OUTPUT_DIR / "raw_validation.json"

REQUIRED_COLUMNS = [
    "Timestamp",
    "Model",
    "Request tokens",
    "Response tokens",
    "Total tokens",
    "Log Type",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA-256 checksum of the raw dataset."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def validate_required_columns(df: pd.DataFrame):
    """Check whether all expected columns exist."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing_columns)
            + f"\n\nColumns actually found:\n{list(df.columns)}"
        )


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print("=" * 70)
    print("BURSTGPT DATA VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check file
    # --------------------------------------------------------

    print("\n[1] Checking dataset file...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{DATA_PATH.resolve()}\n\n"
            f"Place BurstGPT_1.csv inside:\n"
            f"data/BurstGPT_1.csv"
        )

    print(f"Dataset found: {DATA_PATH}")

    # --------------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------------

    print("\n[2] Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # 3. Validate columns
    # --------------------------------------------------------

    print("\n[3] Validating required columns...")

    validate_required_columns(df)

    print("All required columns are present.")

    # --------------------------------------------------------
    # 4. Convert numeric columns
    # --------------------------------------------------------

    print("\n[4] Validating numeric fields...")

    numeric_columns = [
        "Timestamp",
        "Request tokens",
        "Response tokens",
        "Total tokens",
    ]

    numeric_conversion_errors = {}

    for column in numeric_columns:

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        invalid_count = int(converted.isna().sum())

        numeric_conversion_errors[column] = invalid_count

        if invalid_count > 0:
            print(
                f"WARNING: {column} contains "
                f"{invalid_count:,} non-numeric values."
            )
        else:
            print(f"{column}: OK")

        df[column] = converted

    # --------------------------------------------------------
    # 5. Missing values
    # --------------------------------------------------------

    print("\n[5] Checking missing values...")

    missing_values = {}

    for column in df.columns:

        count = int(df[column].isna().sum())

        missing_values[column] = count

        if count > 0:
            print(f"{column}: {count:,} missing")
        else:
            print(f"{column}: 0 missing")

    # --------------------------------------------------------
    # 6. Timestamp validation
    # --------------------------------------------------------

    print("\n[6] Checking timestamps...")

    timestamp_diff = df["Timestamp"].diff()

    timestamp_decreases = int(
        (timestamp_diff.dropna() < 0).sum()
    )

    timestamp_min = int(df["Timestamp"].min())
    timestamp_max = int(df["Timestamp"].max())

    print(f"Minimum timestamp : {timestamp_min:,}")
    print(f"Maximum timestamp : {timestamp_max:,}")
    print(
        f"Timestamp decreases: "
        f"{timestamp_decreases:,}"
    )

    if timestamp_decreases == 0:
        print("Timestamp ordering: OK")
    else:
        print(
            "WARNING: timestamps are not "
            "nondecreasing."
        )

    # --------------------------------------------------------
    # 7. Token arithmetic
    # --------------------------------------------------------

    print("\n[7] Checking token arithmetic...")

    expected_total = (
        df["Request tokens"]
        + df["Response tokens"]
    )

    arithmetic_errors = int(
        (
            df["Total tokens"]
            != expected_total
        ).sum()
    )

    print(
        "Rows where "
        "Total tokens != Request tokens + Response tokens:"
    )
    print(f"  {arithmetic_errors:,}")

    if arithmetic_errors == 0:
        print("Token arithmetic: OK")
    else:
        print(
            "WARNING: token arithmetic "
            "violations detected."
        )

    # --------------------------------------------------------
    # 8. Negative token values
    # --------------------------------------------------------

    print("\n[8] Checking negative token values...")

    negative_counts = {}

    for column in [
        "Request tokens",
        "Response tokens",
        "Total tokens",
    ]:

        count = int(
            (df[column] < 0).sum()
        )

        negative_counts[column] = count

        print(
            f"{column}: "
            f"{count:,} negative values"
        )

    # --------------------------------------------------------
    # 9. Zero-response records
    # --------------------------------------------------------

    print("\n[9] Checking zero-response records...")

    zero_response_count = int(
        (df["Response tokens"] == 0).sum()
    )

    zero_request_count = int(
        (df["Request tokens"] == 0).sum()
    )

    zero_total_count = int(
        (df["Total tokens"] == 0).sum()
    )

    print(
        f"Response tokens == 0 : "
        f"{zero_response_count:,}"
    )

    print(
        f"Request tokens == 0  : "
        f"{zero_request_count:,}"
    )

    print(
        f"Total tokens == 0    : "
        f"{zero_total_count:,}"
    )

    # --------------------------------------------------------
    # 10. Exact duplicate rows
    # --------------------------------------------------------

    print("\n[10] Checking exact duplicate rows...")

    duplicate_mask = df.duplicated(
        keep=False
    )

    duplicate_rows = int(
        duplicate_mask.sum()
    )

    duplicate_groups = int(
        df[duplicate_mask]
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"Rows belonging to duplicate groups: "
        f"{duplicate_rows:,}"
    )

    print(
        f"Unique duplicated row patterns: "
        f"{duplicate_groups:,}"
    )

    print(
        "\nIMPORTANT:"
        "\nDuplicate rows will NOT be removed."
        "\nThe source has no request identifier."
    )

    # --------------------------------------------------------
    # 11. Models
    # --------------------------------------------------------

    print("\n[11] Model distribution...")

    model_counts = (
        df["Model"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    for model, count in model_counts.items():
        print(
            f"  {model}: {count:,}"
        )

    # --------------------------------------------------------
    # 12. Log type distribution
    # --------------------------------------------------------

    print("\n[12] Log Type distribution...")

    log_type_counts = (
        df["Log Type"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    for log_type, count in log_type_counts.items():
        print(
            f"  {log_type}: {count:,}"
        )

    # --------------------------------------------------------
    # 13. Token totals
    # --------------------------------------------------------

    print("\n[13] Token totals...")

    request_token_sum = int(
        df["Request tokens"].sum()
    )

    response_token_sum = int(
        df["Response tokens"].sum()
    )

    total_token_sum = int(
        df["Total tokens"].sum()
    )

    print(
        f"Request tokens : "
        f"{request_token_sum:,}"
    )

    print(
        f"Response tokens: "
        f"{response_token_sum:,}"
    )

    print(
        f"Total tokens   : "
        f"{total_token_sum:,}"
    )

    # --------------------------------------------------------
    # 14. SHA-256
    # --------------------------------------------------------

    print("\n[14] Calculating SHA-256...")

    sha256 = calculate_sha256(DATA_PATH)

    print(f"SHA-256: {sha256}")

    # --------------------------------------------------------
    # 15. Build validation report
    # --------------------------------------------------------

    report = {
        "dataset": {
            "file": str(DATA_PATH),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "sha256": sha256,
        },

        "timestamp": {
            "minimum": timestamp_min,
            "maximum": timestamp_max,
            "decreases": timestamp_decreases,
            "nondecreasing": timestamp_decreases == 0,
        },

        "missing_values": missing_values,

        "numeric_conversion_errors": (
            numeric_conversion_errors
        ),

        "token_validation": {
            "request_tokens_sum": request_token_sum,
            "response_tokens_sum": response_token_sum,
            "total_tokens_sum": total_token_sum,
            "arithmetic_errors": arithmetic_errors,
            "negative_values": negative_counts,
        },

        "zero_response": {
            "zero_response_rows": zero_response_count,
            "zero_request_rows": zero_request_count,
            "zero_total_rows": zero_total_count,
        },

        "duplicates": {
            "rows_in_duplicate_groups": duplicate_rows,
            "duplicate_patterns": duplicate_groups,
            "removed": False,
        },

        "categorical_distribution": {
            "models": model_counts,
            "log_types": log_type_counts,
        },
    }

    # --------------------------------------------------------
    # 16. Save report
    # --------------------------------------------------------

    print("\n[15] Saving validation report...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print(
        f"Report saved to:\n"
        f"{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # 17. Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    if (
        arithmetic_errors == 0
        and timestamp_decreases == 0
        and not any(missing_values.values())
        and not any(
            numeric_conversion_errors.values()
        )
        and not any(
            negative_counts.values()
        )
    ):

        print(
            "\nSTATUS: PASS"
        )

    else:

        print(
            "\nSTATUS: REVIEW REQUIRED"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()