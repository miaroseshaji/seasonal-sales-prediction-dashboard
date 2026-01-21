import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# -----------------------------
# Module 6: Model Training
# -----------------------------

# Load feature engineered data
file_path = r"C:\Users\Admin\Desktop\seasonal_sales_prediction\feature_data.xlsx"
df = pd.read_excel(file_path)

# -----------------------------
# Define features and target
# -----------------------------
X = df[['Year', 'Month']]   # Input features
y = df['Sales']             # Target variable

# -----------------------------
# Split data into train and test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -----------------------------
# Train Linear Regression model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Make predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluate the model
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics")
print("------------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.2f}")

# -----------------------------
# Save trained model (optional)
# -----------------------------
import joblib
model_path = r"C:\Users\Admin\Desktop\seasonal_sales_prediction\sales_model.pkl"
joblib.dump(model, model_path)

print("✅ Trained model saved as sales_model.pkl")
