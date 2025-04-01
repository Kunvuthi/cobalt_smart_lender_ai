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
import shap
from io import BytesIO
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------------------------
# S3 MODEL CONFIG
# ------------------------------------------------------------------------------
S3_BUCKET = "cobalt-lending-ai-data-lake"
S3_MODEL_KEY = "models/xgboost/xgb_model_tree.pkl"
local_model_path = Path("models") / "xgb_model_tree.pkl"

# ------------------------------------------------------------------------------
# OUTPUT HISTORY DIRECTORY
# ------------------------------------------------------------------------------
HISTORY_OUTPUT_DIR = Path("../../data/3-outputs/history")
HISTORY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# FASTAPI INIT
# ------------------------------------------------------------------------------
app = FastAPI(title="Cobalt XGBoost Inference API")
xgb_model = None
explainer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global xgb_model, explainer
    s3 = boto3.client("s3")

    try:
        print(f"[INFO] Downloading model from s3://{S3_BUCKET}/{S3_MODEL_KEY}")
        local_model_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(S3_BUCKET, S3_MODEL_KEY, str(local_model_path))
        xgb_model = joblib.load(local_model_path)
        explainer = shap.TreeExplainer(xgb_model)
        print("[INFO] Model and SHAP Explainer ready.")
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        raise RuntimeError("Failed to load model from S3.")

    yield

app.router.lifespan_context = lifespan

# ------------------------------------------------------------------------------
# Pydantic Schema
# ------------------------------------------------------------------------------
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
        allow_population_by_field_name = True

class BulkInput(BaseModel):
    data: List[Dict]

# ------------------------------------------------------------------------------
# Prediction Helpers
# ------------------------------------------------------------------------------
def predict_proba_df(df: pd.DataFrame) -> np.ndarray:
    return xgb_model.predict_proba(df)[:, 1]

# ------------------------------------------------------------------------------
# SINGLE PREDICTION
# ------------------------------------------------------------------------------
@app.post("/predict")
def predict_single(input_data: SingleInput):
    row = pd.DataFrame([input_data.model_dump(by_alias=True)])
    y_proba = predict_proba_df(row)[0]
    shap_vals = explainer.shap_values(row)[0].tolist()

    return {
        "prob_default": float(y_proba),
        "shap_values": shap_vals,
        "base_value": float(explainer.expected_value),
        "features": row.columns.tolist(),
        "input_row": row.iloc[0].to_dict()
    }

# ------------------------------------------------------------------------------
# BULK PREDICTION (LIMITED TO 10 ROWS)
# ------------------------------------------------------------------------------
@app.post("/predict_bulk_csv")
async def predict_bulk_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        df['prob_default'] = predict_proba_df(df)

        # Replace NaN, inf, -inf to be safe
        df_clean = df.replace([np.inf, -np.inf], np.nan).fillna("null")

        return {"predictions": df_clean.to_dict(orient="records")}
    except Exception as e:
        print(f"[ERROR] Bulk prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk prediction failed: {e}")
    
@app.post("/feature_importance_bulk")
def feature_importance_bulk(data: BulkInput):
    if not data.data:
        raise HTTPException(status_code=400, detail="No data provided.")

    df = pd.DataFrame(data.data)
    try:
        booster = xgb_model.get_booster()
        importance_dict = booster.get_score(importance_type='gain')
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:10]
        return {
            "top_features": [{"feature": k, "importance": v} for k, v in top_features]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature importance computation failed: {e}")
    
# ------------------------------------------------------------------------------
# DEV ENTRY
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("cobalt_fast_api:app", host="0.0.0.0", port=8000, reload=True)
    # uvicorn cobalt_fast_api:app --host 0.0.0.0 --port 8000