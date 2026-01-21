import pandas as pd
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# Module 8: Evaluation & Visualization
# -----------------------------

# Load trained model
model_path = r"C:\Users\Admin\Desktop\seasonal_sales_prediction\sales_model.pkl"
model = joblib.load(model_path)

# Load feature data
data_path = r"C:\Users\Admin\Desktop\seasonal_sales_prediction\feature_data.xlsx"
df = pd.read_excel(data_path)

# -----------------------------
# Prepare data
# -----------------------------
X = df[['Year', 'Month']]
y_actual = df['Sales']

# Predict sales
y_predicted = model.predict(X)

# -----------------------------
# Actual vs Predicted Plot
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(y_actual.values, label='Actual Sales', marker='o')
plt.plot(y_predicted, label='Predicted Sales', marker='x')
plt.title('Actual vs Predicted Sales')
plt.xlabel('Data Points (Time)')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Save comparison data
# -----------------------------
comparison_df = df.copy()
comparison_df['Predicted_Sales'] = y_predicted

output_path = r"C:\Users\Admin\Desktop\seasonal_sales_prediction\actual_vs_predicted.xlsx"
comparison_df.to_excel(output_path, index=False)

print("✅ Evaluation completed and results saved as actual_vs_predicted.xlsx")
 