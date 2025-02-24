import os
import pandas as pd
import numpy as np
import boto3
from io import BytesIO, StringIO

# ------------------------------------------------------------------------------
# S3 CONFIGURATION
# ------------------------------------------------------------------------------
# Replace these with your actual bucket name and object keys.
BUCKET_NAME = "cobalt-lending-ai-data-lake"

# FULL dataset (e.g., gzipped file)
RAW_DATA_KEY_FULL = "dataset/1-raw/LendingClubFullData2007-2020Q3"
CLEAN_DATA_KEY_FULL = "dataset/2-intermediate/full_dataset_cleaned.csv"

# SAMPLE dataset (CSV)
RAW_DATA_KEY_SAMPLE = "dataset/1-raw/100kSampleData"
CLEAN_DATA_KEY_SAMPLE = "dataset/2-intermediate/sample_100k_cleaned.csv"

# ------------------------------------------------------------------------------
# Initialize S3 client
# ------------------------------------------------------------------------------
s3_client = boto3.client("s3")


def drop_columns_with_missing_values(df: pd.DataFrame, threshold_percentage: float = 70.0) -> pd.DataFrame:
    """
    Drops columns that have more than 'threshold_percentage' percent missing values.
    """
    null_counts = df.isnull().sum()
    null_percent = (null_counts / len(df)) * 100
    columns_to_drop = null_percent[null_percent > threshold_percentage].index.tolist()

    print(f"[INFO] Dropping columns with >{threshold_percentage}% missing: {columns_to_drop}")
    df = df.drop(columns=columns_to_drop)
    return df


def load_data_from_s3(use_sample: bool = True) -> pd.DataFrame:
    """
    Loads the dataset from Amazon S3.
    - If `use_sample=True`, loads the sample CSV.
    - If `use_sample=False`, loads the full dataset (gzipped).
    """
    if use_sample:
        key = RAW_DATA_KEY_SAMPLE
        print(f"[INFO] Loading SAMPLE dataset from s3://{BUCKET_NAME}/{key}")
    else:
        key = RAW_DATA_KEY_FULL
        print(f"[INFO] Loading FULL dataset from s3://{BUCKET_NAME}/{key}")

    obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
    body = obj["Body"].read()

    if use_sample:
        # Normal CSV
        df = pd.read_csv(BytesIO(body), low_memory=False)
    else:
        # Gzip file
        df = pd.read_csv(BytesIO(body), low_memory=False)

    return df


def save_data_to_s3(df: pd.DataFrame, use_sample: bool = True):
    """
    Saves the cleaned dataset to S3 with different file paths for sample vs. full.
    """
    if use_sample:
        key = CLEAN_DATA_KEY_SAMPLE
        print(f"[INFO] Saving CLEANED SAMPLE data to s3://{BUCKET_NAME}/{key}")
    else:
        key = CLEAN_DATA_KEY_FULL
        print(f"[INFO] Saving CLEANED FULL data to s3://{BUCKET_NAME}/{key}")

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=csv_buffer.getvalue())
    print("[INFO] Upload complete.\n")


def clean_data_flow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Custom flow:
    1) Drop 'Unnamed: 0.1', 'Unnamed: 0'
    2) Basic info
    3) Fill 'hardship_status' with 'No Hardship' & 'num_accts_ever_120_pd' with mean
    4) Drop columns >70% missing
    5) Drop duplicates
    6) Cap outliers at 95th percentile (loan_amnt)
    7) Return cleaned df
    """

    # 1. Drop known columns
    list_to_drop = ["Unnamed: 0.1", "Unnamed: 0"]
    print(f"[INFO] Dropping columns: {list_to_drop}")
    df_dropped = df.drop(columns=list_to_drop, errors="ignore")

    # 2. Basic info
    print("\n[INFO] DataFrame Info BEFORE cleaning:")
    print(df_dropped.info())
    print("\n[INFO] Missing Value Counts BEFORE cleaning:")
    print(df_dropped.isnull().sum())

    # 3. Fill specific columns
    if "hardship_status" in df_dropped.columns:
        df_dropped["hardship_status"] = df_dropped["hardship_status"].fillna("No Hardship")
        print("[INFO] Filled 'hardship_status' with 'No Hardship'.")

    if "num_accts_ever_120_pd" in df_dropped.columns:
        mean_val = df_dropped["num_accts_ever_120_pd"].mean()
        df_dropped["num_accts_ever_120_pd"].fillna(mean_val, inplace=True)
        print(f"[INFO] Filled 'num_accts_ever_120_pd' with mean: {mean_val}")

    # 4. Drop columns with >70% missing
    df_dropped = drop_columns_with_missing_values(df_dropped, threshold_percentage=70.0)

    # 5. Check & drop duplicates
    before_dupes = len(df_dropped)
    df_dropped.drop_duplicates(inplace=True)
    after_dupes = len(df_dropped)
    print(f"[INFO] Duplicates removed: {before_dupes - after_dupes}")

    # 6. Cap outliers at 95% tile (example: 'loan_amnt')
    if "loan_amnt" in df_dropped.columns:
        q95 = df_dropped["loan_amnt"].quantile(0.95)
        df_dropped["loan_amnt"] = np.where(df_dropped["loan_amnt"] > q95, q95, df_dropped["loan_amnt"])
        print(f"[INFO] Capped 'loan_amnt' at 95th percentile value: {q95}")

    print("\n[INFO] DataFrame Info AFTER cleaning:")
    print(df_dropped.info())
    print("\n[INFO] Missing Value Counts AFTER cleaning:")
    print(df_dropped.isnull().sum())

    print("[INFO] Clean data flow complete.")
    return df_dropped


def main(use_sample: bool = True):
    """
    Main orchestration:
    1. Load data from S3
    2. Clean it with 'clean_data_flow'
    3. Save cleaned data back to S3
    """
    print("\n[INFO] Starting 'main' process...")
    df = load_data_from_s3(use_sample=use_sample)

    cleaned_df = clean_data_flow(df)

    save_data_to_s3(cleaned_df, use_sample=use_sample)
    print("[INFO] Done.\n")


if __name__ == "__main__":
    """
    Example usage from terminal:
      python clean_data.py        # uses sample dataset
      python clean_data.py full   # uses full dataset
    """
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "full":
        main(use_sample=False)
    else:
        main(use_sample=True)