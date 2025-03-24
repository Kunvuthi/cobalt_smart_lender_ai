"""
This script contains the feature engineering steps for the Lending Club dataset.
"""

import os
import pandas as pd
import numpy as np
import boto3
from io import BytesIO, StringIO

# ------------------------------------------------------------------------------
# S3 CONFIGURATION
# ------------------------------------------------------------------------------
# Bucket name and object keys.
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