import pandas as pd

# -----------------------------
# Module 2: Data Cleaning
# -----------------------------

# Load dataset
file_path = r"C:\Users\Admin\Desktop\seasonal_sales_prediction\Online Retail-Copy1.xlsx"
df = pd.read_excel(file_path)

print("Original Data Shape:", df.shape)

# -----------------------------
# 1. Remove missing values
# -----------------------------
df.dropna(inplace=True)

# -----------------------------
# 2. Remove negative or zero quantities
# -----------------------------
df = df[df['Quantity'] > 0]

# -----------------------------
# 3. Convert InvoiceDate to datetime
# -----------------------------
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print("Cleaned Data Shape:", df.shape)

# -----------------------------
# 4. Save cleaned data (IMPORTANT)
# -----------------------------
cleaned_file_path = r"C:\Users\Admin\Desktop\seasonal_sales_prediction\cleaned_data.xlsx"
df.to_excel(cleaned_file_path, index=False)

print("✅ Cleaned data saved as cleaned_data.xlsx")
