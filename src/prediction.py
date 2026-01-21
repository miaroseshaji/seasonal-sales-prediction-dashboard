import pandas as pd
import joblib

# -----------------------------
# Module 7: Sales Prediction
# -----------------------------

# Load trained model
model_path = r"C:\Users\Admin\Desktop\seasonal_sales_prediction\sales_model.pkl"
model = joblib.load(model_path)

# Load feature data (for reference)
data_path = r"C:\Users\Admin\Desktop\seasonal_sales_prediction\feature_data.xlsx"
df = pd.read_excel(data_path)

# -----------------------------
# Predict future months
# -----------------------------

# Create future data (next 6 months example)
future_data = pd.DataFrame({
    'Year': [2024, 2024, 2024, 2024, 2024, 2024],
    'Month': [1, 2, 3, 4, 5, 6]
})

# Predict sales
future_data['Predicted_Sales'] = model.predict(future_data)

print("Future Sales Prediction:")
print(future_data)

# -----------------------------
# Save predictions
# -----------------------------
output_path = r"C:\Users\Admin\Desktop\seasonal_sales_prediction\future_sales_prediction.xlsx"
future_data.to_excel(output_path, index=False)

print("✅ Future sales predictions saved as future_sales_prediction.xlsx")
