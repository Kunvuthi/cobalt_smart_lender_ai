import streamlit as st
import pandas as pd
import numpy as np
import requests
import shap
import matplotlib.pyplot as plt
import json

# BACKEND API CONFIG
API_URL = "http://cobalt-lender-api:8000"  # Change to your EC2 URL when deployed

# MODEL INPUT COLUMNS
NUMERIC_COLS = [
    "loan_amnt", "term", "installment", "fico_range_low", "last_fico_range_high",
    "open_il_12m", "open_il_24m", "max_bal_bc", "num_rev_accts",
    "pub_rec_bankruptcies", "emp_length_num", "earliest_cr_line_days"
]

DUMMY_COLS = [
    "grade_E",
    "home_ownership_MORTGAGE",
    "verification_status_Verified",
    "application_type_Joint_App",
    "hardship_status_BROKEN",
    "hardship_status_COMPLETE",
    "hardship_status_COMPLETED",
    "hardship_status_No_Hardship",
]

# All input columns (used for ordering)
ALL_COLS = NUMERIC_COLS + DUMMY_COLS


# UI TITLE & MODE SELECTION
st.set_page_config(page_title="Cobalt Loan Default Prediction", layout="wide")
st.title("💳 Loan Default Risk Predictor")

menu = st.sidebar.radio("Select Mode", ["🔍 Single Prediction", "📤 Bulk Prediction + SHAP"])

# --------------------------------------------------------------------------------------------------
# 🔍 SINGLE PREDICTION MODE
# --------------------------------------------------------------------------------------------------
if menu == "🔍 Single Prediction":
    st.subheader("🔹 Enter loan details for a single borrower")

    col1, col2 = st.columns(2)
    input_data = {}

    with col1:
        input_data["loan_amnt"] = st.number_input("Loan Amount", value=10000.0, min_value=0.0)
        input_data["term"] = st.selectbox("Term (months)", [36, 60], index=0)
        input_data["installment"] = st.number_input("Installment", value=300.0)
        input_data["fico_range_low"] = st.number_input("FICO Range Low", value=660.0)
        input_data["last_fico_range_high"] = st.number_input("Last FICO High", value=700.0)
        input_data["open_il_12m"] = st.number_input("Open IL Last 12m", value=1.0)
        input_data["open_il_24m"] = st.number_input("Open IL Last 24m", value=2.0)

    with col2:
        input_data["max_bal_bc"] = st.number_input("Max Balance on Bank Card", value=2000.0)
        input_data["num_rev_accts"] = st.number_input("Number of Revolving Accounts", value=10.0)
        input_data["pub_rec_bankruptcies"] = st.number_input("Bankruptcies", value=0.0)
        input_data["emp_length_num"] = st.number_input("Employment Length (years)", value=3.0)
        input_data["earliest_cr_line_days"] = st.number_input("Days Since First Credit Line", value=4000.0)

        # Dummy / categorical checkboxes
        input_data["grade_E"] = int(st.checkbox("Grade E"))
        input_data["home_ownership_MORTGAGE"] = int(st.checkbox("Home Ownership: Mortgage"))
        input_data["verification_status_Verified"] = int(st.checkbox("Verified Status"))
        input_data["application_type_Joint_App"] = int(st.checkbox("Joint Application"))

        hardship = st.selectbox("Hardship Status", ["ACTIVE", "BROKEN", "COMPLETE", "COMPLETED", "No_Hardship"])
        for status in ["BROKEN", "COMPLETE", "COMPLETED", "No_Hardship"]:
            key = f"hardship_status_{status}"
            input_data[key] = 1 if hardship == status else 0

    key_renames = {
        "application_type_Joint_App": "application_type_Joint App",
        "hardship_status_No_Hardship": "hardship_status_No Hardship",
    }
    for old, new in key_renames.items():
        if old in input_data:
            input_data[new] = input_data.pop(old)

    if st.button("🚀 Predict Default Risk"):
        res = requests.post(f"{API_URL}/predict", json=input_data)

        try:
            res.raise_for_status()
            response_json = res.json()

            prob = response_json["prob_default"]
            shap_values = response_json["shap_values"]
            base_value = response_json["base_value"]
            features = response_json["features"]
            input_row = response_json["input_row"]

            st.success(f"Estimated Default Probability: {prob:.2%}")

            # SHAP Explanation
            st.subheader("🔍 SHAP Explanation")

            shap_explanation = shap.Explanation(
                values=np.array(shap_values),
                base_values=base_value,
                data=np.array([input_row[f] for f in features]),
                feature_names=features
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_explanation, max_display=10, show=False)
            plt.tight_layout()

            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.text(f"Status Code: {res.status_code}")
            st.text(f"Response: {res.text}")

# --------------------------------------------------------------------------------------------------
# 📤 BULK PREDICTION MODE
# --------------------------------------------------------------------------------------------------
elif menu == "📤 Bulk Prediction + SHAP":
    st.subheader("📂 Upload CSV for Bulk Inference")

    uploaded_file = st.file_uploader("Upload CSV with required columns", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("✅ Uploaded Data Preview:", df.head())
        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")
            st.stop()

        if st.button("🚀 Run Bulk Prediction"):
            try:
                # Send CSV file to FastAPI backend
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                response = requests.post(f"{API_URL}/predict_bulk_csv", files=files)
                response.raise_for_status()
                results = response.json()["predictions"]

                # Convert result to DataFrame
                df_result = pd.DataFrame(results)

                # Coerce columns to numeric where possible (NaNs allowed)
                df_result = df_result.apply(pd.to_numeric, errors="coerce")

                st.subheader("📈 Prediction Results")
                st.dataframe(df_result)

                # Download button
                st.download_button("📥 Download Results", df_result.to_csv(index=False), "bulk_predictions.csv")

                # Feature Importance Visualization
                st.subheader("📊 Feature Importance (Top 10)")

                importance_resp = requests.post(f"{API_URL}/feature_importance_bulk", json={"data": results})
                importance_resp.raise_for_status()
                importance_data = importance_resp.json()["top_features"]

                features = [item["feature"] for item in importance_data]
                importances = [item["importance"] for item in importance_data]

                fig, ax = plt.subplots()
                ax.barh(features[::-1], importances[::-1])
                ax.set_xlabel("Importance (gain)")
                ax.set_title("Top 10 Important Features")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Prediction or Feature Importance failed: {e}")