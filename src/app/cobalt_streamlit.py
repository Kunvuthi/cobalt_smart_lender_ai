import streamlit as st
import pandas as pd
import numpy as np
import requests
import shap
import matplotlib.pyplot as plt
import json

# BACKEND API CONFIG
API_URL = "http://localhost:8000"  # Change to your EC2 URL when deployed

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
        df = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded Data Preview:", df.head())

        if st.button("🚀 Run Bulk Prediction"):
            try:
                # Send CSV file to FastAPI
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(f"{API_URL}/predict_bulk_csv", files={"file": uploaded_file})
                results = response.json()["predictions"]

                df_result = pd.DataFrame(results)
                st.subheader("📈 Prediction Results")
                st.dataframe(df_result)

                # Download option
                st.download_button("📥 Download Results", df_result.to_csv(index=False), "bulk_predictions.csv")

                # SHAP Bulk
                st.subheader("🔍 SHAP Visualizer")
                shap_resp = requests.post(f"{API_URL}/shap_bulk", json={"data": results})
                shap_json = shap_resp.json()

                all_shap_vals = np.array(shap_json["shap_values"])
                base_val = shap_json["base_value"]
                features = shap_json["features"]
                inputs = pd.DataFrame(shap_json["inputs"])

                selected_index = st.slider("Select row to explain", 0, len(inputs) - 1, 0)
                st.json(inputs.iloc[selected_index].to_dict())

                # Interactive force plot
                explainer = shap.Explanation(
                    values=all_shap_vals[selected_index],
                    base_values=base_val,
                    data=inputs.iloc[selected_index].values,
                    feature_names=features
                )

                st.set_option('deprecation.showPyplotGlobalUse', False)
                shap.plots.waterfall(explainer, max_display=10)
                st.pyplot()

            except Exception as e:
                st.error(f"Prediction or SHAP failed: {e}")