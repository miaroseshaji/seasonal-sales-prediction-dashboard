import pandas as pd

# -----------------------------
# Module 1: Data Collection
# -----------------------------

# Load the dataset
file_path =  r"C:\Users\Admin\Desktop\seasonal_sales_prediction\Online Retail-Copy1.xlsx"
df = pd.read_excel(file_path)

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

print("\n-----------------------------------\n")

# Display dataset information
print("Dataset Information:")
print(df.info())

print("\n-----------------------------------\n")

# Display basic statistics
print("Statistical Summary:")
print(df.describe())

print("\n-----------------------------------\n")

# Display column names
print("Column Names:")
print(df.columns.tolist())
