"""
This script contains the feature engineering steps for the Lending Club dataset.
"""

import os
import pandas as pd
import numpy as np
import boto3
from io import BytesIO, StringIO
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------------------------
# S3 CONFIGURATION
# ------------------------------------------------------------------------------
BUCKET_NAME = "cobalt-lending-ai-data-lake"
RAW_DATA_KEY_FULL = "dataset/2-intermediate/full_dataset_cleaned_01.csv"
CLEAN_DATA_KEY_TREE = "dataset/2-intermediate/full_dataset_cleaned_02_tree.csv"
CLEAN_DATA_KEY_NN = "dataset/2-intermediate/full_dataset_cleaned_02_nn.csv"

s3_client = boto3.client("s3")

def load_data_from_s3() -> pd.DataFrame:
    """
    Downloads the 'cleaned_01' dataset from S3 and loads it into a DataFrame.
    """
    print(f"[INFO] Loading CLEANED v1 dataset from s3://{BUCKET_NAME}/{RAW_DATA_KEY_FULL}")
    obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=RAW_DATA_KEY_FULL)
    body = obj["Body"].read()
    df = pd.read_csv(BytesIO(body), low_memory=False)
    return df

def save_data_to_s3(df: pd.DataFrame, key: str):
    """
    Saves a given DataFrame as CSV back to the specified key in the S3 bucket.
    """
    print(f"[INFO] Saving data to s3://{BUCKET_NAME}/{key}")
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=csv_buffer.getvalue())
    print("[INFO] Upload complete.\n")

def clean_lending_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleaning steps:
      1) Drop known leakage/useless columns.
      2) Drop rows with too many missing values.
      3) Convert emp_length to numeric.
      4) Convert revol_util from % to float ratio.
      5) Convert earliest_cr_line to # days from present.
      6) Create 'loan_default' from 'loan_status', then drop 'loan_status'.
    """
    print(f"[INFO] Cleaning dataset with shape: {df.shape}")

    # 1) Drop known leakage/useless columns
    leakage_cols = ['recoveries', 'collection_recovery_fee', 'debt_settlement_flag']
    useless_cols = [
        'id', 'url', 'title', 'zip_code', 'addr_state', 'emp_title', 'issue_d',
        'initial_list_status', 'hardship_flag', 'sub_grade', 'next_pymnt_d',
        'last_credit_pull_d', 'pymnt_plan'
    ]
    df.drop(columns=leakage_cols + useless_cols, inplace=True, errors='ignore')

    # 2) Drop rows with more than 20 missing columns
    df.dropna(thresh=df.shape[1] - 20, inplace=True)

    # 3) Convert emp_length => numeric (emp_length_num)
    df['emp_length'] = df['emp_length'].replace('< 1 year', '0')
    df['emp_length_num'] = pd.to_numeric(df['emp_length'].str.extract(r'(\d+)')[0], errors='coerce')
    df.drop(columns=['emp_length'], inplace=True)

    # 4) Convert revol_util (string percent) => float ratio
    df['revol_util'] = df['revol_util'].str.replace('%', '', regex=False).astype(float) / 100

    # 5) Convert earliest_cr_line => # of days from 'today'
    today = datetime.today()
    date_col = 'earliest_cr_line'
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], format='%b-%Y', errors='coerce')
        df[date_col + '_days'] = (today - df[date_col]).dt.days
        df.drop(columns=[date_col], inplace=True, errors='ignore')

    # 6) Create 'loan_default' from 'loan_status' then drop 'loan_status'
    loan_status_map = {
        'Fully Paid': 0,
        'Current': 0,
        'Issued': 0,
        'In Grace Period': 0,
        'Late (16-30 days)': 0,
        'Late (31-120 days)': 1,
        'Charged Off': 1,
        'Default': 1
    }
    if 'loan_status' in df.columns:
        df['loan_default'] = df['loan_status'].map(loan_status_map)
        df.drop(columns=['loan_status'], inplace=True, errors='ignore')

    print(f"[INFO] Done Cleaning dataset with shape: {df.shape}")

    return df

def feature_engineer_lending_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform feature engineering to produce two DataFrames:
      1) df_log_tree  => One-hot encoded for tree-based models
      2) df_log_nn    => Imputed + label-encoded for neural networks

      Steps:
       a) Log-transform skewed numeric columns in both versions
       b) For df_log_tree, do get_dummies(...) on categorical columns
       c) For df_log_nn, create missing indicators + median impute
       d) Label encode any object columns for the NN dataset
    """
    print(f"[INFO] Feature engineering on dataset. Info below:")
    print(df.info())

    columns_log = [
        'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'installment', 'annual_inc', 'dti',
        'fico_range_low', 'fico_range_high', 'mths_since_last_delinq', 'open_acc', 'total_acc',
        'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
        'last_pymnt_amnt', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim',
        'earliest_cr_line_days', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy',
        'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc',
        'mths_since_recent_bc', 'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
        'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl',
        'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
        'num_tl_op_past_12m', 'pub_rec_bankruptcies', 'tot_hi_cred_lim', 'total_bal_ex_mort',
        'total_bc_limit', 'total_il_high_credit_limit', 'revol_util'
    ]

    # Copy to apply log transform
    df_log = df.copy()
    for col in columns_log:
        if col in df_log.columns:
            # Skip entirely missing or non-positive columns
            if df_log[col].notnull().sum() == 0 or (df_log[col].dropna() <= 0).all():
                continue
            df_log[col] = df_log[col].apply(lambda x: np.log1p(x) if pd.notnull(x) and x > 0 else x)

    # ---------------- TREE DATASET ----------------
    df_log_tree = pd.get_dummies(
        df_log.copy(),
        columns=['grade', 'home_ownership', 'verification_status', 'purpose',
                 'application_type', 'hardship_status'],
        drop_first=True
    )

    # ---------------- NN DATASET ----------------
    df_log_nn = df_log.copy()

    null_counts = df_log_nn.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0].index.tolist()

    # For numeric columns that are missing, median impute + missing indicator
    for col in columns_with_nulls:
        # Skip 'dti' special logic + skip non-numeric columns
        if col == 'dti' or not np.issubdtype(df_log_nn[col].dtype, np.number):
            continue
        df_log_nn[col + '_NA'] = df_log_nn[col].isnull().astype(int)
        median_value = df_log_nn[col].median()
        df_log_nn[col] = df_log_nn[col].fillna(median_value)

    # Special handling for dti
    df_log_nn['no_income'] = ((df_log_nn['annual_inc'].isnull()) | (df_log_nn['annual_inc'] == 0)).astype(int)
    df_log_nn['dti_NA'] = df_log['dti'].isnull().astype(int)
    df_log_nn['dti'] = df_log_nn['dti'].fillna(df_log_nn['dti'].median())

    # Label encode any remaining object columns
    object_cols = df_log_nn.select_dtypes(include=['object', 'category']).columns.tolist()
    label_encoders = {}
    for col in object_cols:
        le = LabelEncoder()
        df_log_nn[col] = df_log_nn[col].astype(str).fillna("missing")
        df_log_nn[col] = le.fit_transform(df_log_nn[col])
        label_encoders[col] = le

    print(f"[INFO] Done Feature engineering for tree dataset. Info below:")
    print(df_log_tree.info())

    print(f"[INFO] Done Feature engineering for NN dataset. Info below:")
    print(df_log_nn.info())

    return df_log_tree, df_log_nn

def main():
    # 1) Load data from S3
    df_raw = load_data_from_s3()

    # 2) Clean data (drop leaks/useless, create loan_default, etc.)
    df_cleaned = clean_lending_data(df_raw)

    # 3) Feature engineer for tree & nn
    df_tree_ready, df_nn_ready = feature_engineer_lending_data(df_cleaned)

    # 4) Diagnostics: check any NaN counts in each dataset
    print("[DIAGNOSTIC] NaN values in tree dataset:")
    print(df_tree_ready.isnull().sum().sort_values(ascending=False).head())
    print("[DIAGNOSTIC] NaN values in NN dataset:")
    print(df_nn_ready.isnull().sum().sort_values(ascending=False).head())

    # 5) Save the two final versions to S3
    save_data_to_s3(df_tree_ready, CLEAN_DATA_KEY_TREE)
    save_data_to_s3(df_nn_ready, CLEAN_DATA_KEY_NN)

if __name__ == "__main__":
    main()