import os
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Dict
import boto3
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from io import BytesIO
import shap
from pathlib import Path

# --------------------------------------------------------------------------
# S3 CONFIGURATION
# --------------------------------------------------------------------------
S3_BUCKET = "cobalt-lending-ai-data-lake"
S3_MODEL_KEY = "models/xgboost/xgb_model_tree.pkl"
local_model_path = Path("models") / "xgb_model_tree.pkl"

# --------------------------------------------------------------------------
# APP INIT
# --------------------------------------------------------------------------

xgb_model = None
explainer = None  # SHAP explainer

@asynccontextmanager
async def lifespan(app: FastAPI):
    global xgb_model, explainer

    s3 = boto3.client("s3")

    try:
        # Ensure local directory exists
        local_model_path.parent.mkdir(parents=True, exist_ok=True)

        # Download model from S3
        print(f"[INFO] Downloading model from s3://{S3_BUCKET}/{S3_MODEL_KEY} to {local_model_path}")
        s3.download_file(S3_BUCKET, S3_MODEL_KEY, str(local_model_path))
        print("[INFO] Download complete.")

        # Load model
        xgb_model = joblib.load(str(local_model_path))
        explainer = shap.TreeExplainer(xgb_model)
        print("[INFO] Model and SHAP Explainer loaded.")

    except Exception as e:
        print(f"[ERROR] Failed to download or load model: {e}")
        raise RuntimeError("Startup failed — could not load model from S3.") from e

    yield

app = FastAPI(title="XGBoost Loan Risk Inference API", lifespan=lifespan)

# --------------------------------------------------------------------------
# COLUMNS TO LOG TRANSFORM
# --------------------------------------------------------------------------
LOG_COLS = [
    "loan_amnt", "term", "installment", "fico_range_low", "last_fico_range_high",
    "open_il_12m", "open_il_24m", "max_bal_bc", "num_rev_accts",
    "pub_rec_bankruptcies", "emp_length_num", "earliest_cr_line_days"
]

# --------------------------------------------------------------------------
# Pydantic Input Schema
# --------------------------------------------------------------------------
class SingleInput(BaseModel):
    loan_amnt: float
    term: float
    installment: float
    fico_range_low: float
    last_fico_range_high: float
    open_il_12m: float
    open_il_24m: float
    max_bal_bc: float
    num_rev_accts: float
    pub_rec_bankruptcies: float
    emp_length_num: float
    earliest_cr_line_days: float
    grade_E: int
    home_ownership_MORTGAGE: int
    verification_status_Verified: int
    application_type_Joint_App: int = Field(alias="application_type_Joint App")
    hardship_status_BROKEN: int
    hardship_status_COMPLETE: int
    hardship_status_COMPLETED: int
    hardship_status_No_Hardship: int = Field(alias="hardship_status_No Hardship")

    class Config:
        allow_population_by_field_name = True  # allows using either alias or field name

class BulkInput(BaseModel):
    data: List[Dict]

# --------------------------------------------------------------------------
# HELPER: Log-transform specific columns
# --------------------------------------------------------------------------
def transform_row(row: Dict) -> Dict:
    return {
        col: float(np.log1p(val)) if col in LOG_COLS and val > 0 else val
        for col, val in row.items()
    }

def predict_proba_df(df: pd.DataFrame) -> np.ndarray:
    return xgb_model.predict_proba(df)[:, 1]

# --------------------------------------------------------------------------
# ENDPOINT: Single Prediction
# --------------------------------------------------------------------------
@app.post("/predict")
def predict_single(input_data: SingleInput):
    row_dict = transform_row(input_data.model_dump(by_alias=True))
    df = pd.DataFrame([row_dict])

    # Predict Probability
    y_proba = predict_proba_df(df)[0]

    # SHAP Values for interpretability
    shap_values = explainer.shap_values(df)[0].tolist()
    base_value = float(explainer.expected_value)

    # Return combined prediction and SHAP details
    return {
        "prob_default": float(y_proba),
        "shap_values": shap_values,
        "base_value": base_value,
        "features": list(df.columns),
        "input_row": row_dict
    }

# --------------------------------------------------------------------------
# ENDPOINT: Bulk Prediction via CSV Upload
# --------------------------------------------------------------------------
@app.post("/predict_bulk_csv")
async def predict_bulk_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")

    missing_cols = [col for col in SingleInput.model_fields if col not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")

    df_transformed = df.apply(transform_row, axis=1, result_type='expand')
    y_proba = predict_proba_df(df_transformed)
    df["prob_default"] = y_proba

    return {"predictions": df.to_dict(orient="records")}

# --------------------------------------------------------------------------
# ENDPOINT: SHAP for Single Row
# --------------------------------------------------------------------------
@app.post("/shap_single")
def shap_single(input_data: SingleInput):
    row_dict = transform_row(input_data.model_dump(by_alias=True))
    X = pd.DataFrame([row_dict])
    shap_values = explainer.shap_values(X)[0].tolist()

    return {
        "shap_values": shap_values,
        "base_value": float(explainer.expected_value),
        "features": list(X.columns),
        "input_row": row_dict
    }

# --------------------------------------------------------------------------
# ENDPOINT: SHAP for Bulk JSON Input
# --------------------------------------------------------------------------
@app.post("/shap_bulk")
def shap_bulk(bulk_data: BulkInput):
    if not bulk_data.data:
        raise HTTPException(status_code=400, detail="No data provided.")

    transformed = [transform_row(row) for row in bulk_data.data]
    X = pd.DataFrame(transformed)
    shap_vals = explainer.shap_values(X).tolist()

    return {
        "shap_values": shap_vals,
        "base_value": float(explainer.expected_value),
        "features": list(X.columns),
        "inputs": transformed
    }

# --------------------------------------------------------------------------
# LOCAL DEBUG ENTRY POINT
# --------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("cobalt_fast_api:app", host="0.0.0.0", port=8000, reload=True)
    # uvicorn cobalt_fast_api:app --host 0.0.0.0 --port 8000