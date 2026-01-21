import pandas as pd

# -----------------------------
# Module 5: Feature Engineering
# -----------------------------

# Load cleaned dataset
file_path = r"C:\Users\Admin\Desktop\seasonal_sales_prediction\cleaned_data.xlsx"
df = pd.read_excel(file_path)

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# -----------------------------
# Create new features
# -----------------------------

# Total sales per record
df['Sales'] = df['Quantity'] * df['UnitPrice']

# Time-based features
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day

# -----------------------------
# Aggregate monthly sales (ML-ready data)
# -----------------------------
monthly_features = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()

print("Feature Engineered Data:")
print(monthly_features.head())

# -----------------------------
# Save feature engineered data
# -----------------------------
output_path = r"C:\Users\Admin\Desktop\seasonal_sales_prediction\feature_data.xlsx"
monthly_features.to_excel(output_path, index=False)

print("✅ Feature engineered data saved as feature_data.xlsx")
