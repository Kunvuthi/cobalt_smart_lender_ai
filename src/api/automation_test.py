import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Configuration
original_data_path = Path("../../data/2-intermediate/df_out_dsif3_tree.csv")
automation_input_path = Path("../../data/data-input-automation/test_sample.csv")
output_file_path = Path("../../data/3-outputs/latest_output.csv")

# Load original CSV
df = pd.read_csv(original_data_path)

# Select relevant columns for X
X_cols = [
    'loan_amnt', 'term', 'installment', 'fico_range_low', 'last_fico_range_high',
    'open_il_12m', 'open_il_24m', 'max_bal_bc', 'num_rev_accts', 'pub_rec_bankruptcies',
    'emp_length_num', 'earliest_cr_line_days', 'grade_E', 'home_ownership_MORTGAGE',
    'verification_status_Verified', 'application_type_Joint App', 'hardship_status_BROKEN',
    'hardship_status_COMPLETE', 'hardship_status_COMPLETED', 'hardship_status_No Hardship'
]

X = df[X_cols]
y = df['loan_default']

# Take a 10-row sample
_, X_test, _, y_test = train_test_split(X, y, test_size=10, random_state=42)

# Save sample without label
X_test.to_csv(automation_input_path, index=False)
print(f"Test input saved to: {automation_input_path}")

# Wait for automated prediction (manual check needed after running this script)
print("Waiting for predictions to be generated automatically...")

# Compare manually later:
print("\nActual Labels (y_test):")
print(y_test.reset_index(drop=True))

print(f"\nCheck predictions at: {output_file_path}")