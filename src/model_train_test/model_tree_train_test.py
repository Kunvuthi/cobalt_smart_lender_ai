import os
import sys
import joblib
import json
import boto3
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from io import BytesIO, StringIO
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# AWS S3 config
BUCKET_NAME = "cobalt-lending-ai-data-lake"
INPUT_KEY = "dataset/2-intermediate/full_dataset_cleaned_02_tree.csv"  # Where your CSV is stored
OUTPUT_PATH = "models/xgboost/"  # Where you'll upload artifacts
BEST_MODEL_FILENAME = "xgb_model_tree.pkl"
FEATURES_FILENAME = "selected_features_tree.txt"
METRICS_JSON = "metrics.json"

# Optional: use environment variables for credentials or rely on IAM roles if on EC2
s3_client = boto3.client("s3")


def download_csv_from_s3(bucket, key):
    """
    Downloads a CSV file from S3 and returns it as a Pandas DataFrame.
    """
    logging.info(f"Downloading data from s3://{bucket}/{key}")
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = obj['Body']
    df = pd.read_csv(body, low_memory=False)
    logging.info(f"Data shape: {df.shape}")
    return df

def upload_file_to_s3(local_path, bucket, key):
    """
    Uploads a local file to S3.
    """
    logging.info(f"Uploading {local_path} to s3://{bucket}/{key}")
    s3_client.upload_file(Filename=local_path, Bucket=bucket, Key=key)
    logging.info("Upload complete.")

def upload_bytes_to_s3(data_bytes, bucket, key):
    """
    Uploads in-memory bytes (e.g., a plot or JSON) to S3.
    """
    logging.info(f"Uploading to s3://{bucket}/{key}")
    s3_client.put_object(Bucket=bucket, Key=key, Body=data_bytes)
    logging.info("Upload complete.")

def save_plot_to_s3(fig, bucket, key):
    """
    Saves a Matplotlib figure to S3 in PNG format without writing to local disk.
    """
    png_buffer = BytesIO()
    fig.savefig(png_buffer, format='png')
    png_buffer.seek(0)
    upload_bytes_to_s3(png_buffer, bucket, key)

def main():
    # --------------------------
    # 1. Download the CSV from S3
    # --------------------------
    df_tree = download_csv_from_s3(BUCKET_NAME, INPUT_KEY)

    # --------------------------
    # 2. Remove known leakage columns
    # --------------------------
    leakage_cols = [
        'total_rec_late_fee','total_rec_prncp','out_prncp','last_pymnt_amnt','last_pymnt_d',
        'funded_amnt_inv','funded_amnt','out_prncp_inv','total_pymnt','total_pymnt_inv',
        'last_pymnt_d_days','last_credit_pull_d_days','issue_d_days','total_rec_int'
    ]
    df_tree.drop(columns=leakage_cols, inplace=True, errors='ignore')

    # --------------------------
    # 3. Train/Test Split
    # --------------------------
    X = df_tree.drop('loan_default', axis=1)
    y = df_tree['loan_default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=22
    )
    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # (Optional) If you want SMOTE for imbalance, do it here:
    # from imblearn.over_sampling import SMOTE
    # sm = SMOTE(random_state=42)
    # X_train, y_train = sm.fit_resample(X_train, y_train)

    # --------------------------
    # 4. Calculate scale_pos_weight (if heavily imbalanced)
    # --------------------------
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    logging.info(f"scale_pos_weight={scale_pos_weight:.4f}")

    # --------------------------
    # 5. Feature Selection (RFE to get EXACT 20)
    # --------------------------
    base_model = XGBClassifier(
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    rfe = RFE(estimator=base_model, n_features_to_select=20, step=1)
    rfe.fit(X_train, y_train)

    selected_features = X_train.columns[rfe.support_].tolist()
    logging.info(f"Selected {len(selected_features)} features: {selected_features}")

    # Subset training/test data to these features
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # --------------------------
    # 6. Hyperparameter Tuning with RandomizedSearchCV
    # --------------------------
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    xgb_base = XGBClassifier(
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        random_state=78
    )

    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.5, 0.8, 1.0],
        'gamma': [0, 1, 5],
    }

    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=20,
        scoring='roc_auc',
        cv=StratifiedKFold(3),
        verbose=1,
        n_jobs=-1,
        random_state=22
    )

    random_search.fit(X_train_sel, y_train)

    logging.info(f"Best score (AUC): {random_search.best_score_}")
    logging.info(f"Best params: {random_search.best_params_}")

    best_model_tree = random_search.best_estimator_

    # --------------------------
    # 7. Final Evaluation on Test Set
    # --------------------------
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

    y_pred_test = best_model_tree.predict(X_test_sel)
    y_proba_test = best_model_tree.predict_proba(X_test_sel)[:, 1]

    clf_report = classification_report(y_test, y_pred_test, output_dict=True)
    auc_test = roc_auc_score(y_test, y_proba_test)
    cm = confusion_matrix(y_test, y_pred_test)

    logging.info(f"Classification Report:\n {classification_report(y_test, y_pred_test)}")
    logging.info(f"ROC AUC: {auc_test:.4f}")

    # --------------------------
    # 8. Plot & Save Confusion Matrix
    # --------------------------
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")

    # Save confusion matrix plot to S3
    plot_cm_key = os.path.join(OUTPUT_PATH, "confusion_matrix.png")
    save_plot_to_s3(fig_cm, BUCKET_NAME, plot_cm_key)

    # --------------------------
    # 9. Plot & Save Feature Importances
    # --------------------------
    importances = best_model_tree.feature_importances_
    feat_df = pd.DataFrame({'Feature': X_train_sel.columns, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    top_feats = feat_df.head(10)

    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
    ax_imp.barh(top_feats['Feature'][::-1], top_feats['Importance'][::-1], color='skyblue')
    ax_imp.set_xlabel("Feature Importance (Gain)")
    ax_imp.set_title("Top 10 Most Important Features")
    plt.tight_layout()

    plot_importance_key = os.path.join(OUTPUT_PATH, "feature_importance.png")
    save_plot_to_s3(fig_imp, BUCKET_NAME, plot_importance_key)

    # --------------------------
    # 10. Save Model Locally & Upload to S3
    # --------------------------
    os.makedirs("models", exist_ok=True)
    local_model_path = os.path.join("models", BEST_MODEL_FILENAME)
    joblib.dump(best_model_tree, local_model_path)
    model_s3_key = os.path.join(OUTPUT_PATH, BEST_MODEL_FILENAME)
    upload_file_to_s3(local_model_path, BUCKET_NAME, model_s3_key)

    # --------------------------
    # 11. Save Selected Features
    # --------------------------
    local_feat_path = os.path.join("models", "selected_features_tree.txt")
    with open(local_feat_path, "w") as f:
        for item in selected_features:
            f.write("%s\n" % item)
        f.write("\n# Features selected via RFE + XGBoost hyperparam search.\n")
    feats_s3_key = os.path.join(OUTPUT_PATH, FEATURES_FILENAME)
    upload_file_to_s3(local_feat_path, BUCKET_NAME, feats_s3_key)

    # --------------------------
    # 12. Log Metrics to JSON
    # --------------------------
    metrics_dict = {
        "auc": auc_test,
        "classification_report": clf_report,
        "best_params": random_search.best_params_
    }
    metrics_json_str = json.dumps(metrics_dict, indent=2)
    metrics_s3_key = os.path.join(OUTPUT_PATH, METRICS_JSON)
    upload_bytes_to_s3(metrics_json_str.encode('utf-8'), BUCKET_NAME, metrics_s3_key)


if __name__ == "__main__":
    main()