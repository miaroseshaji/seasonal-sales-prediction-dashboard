import pandas as pd

# -----------------------------
# Module 4: Statistical Analysis
# -----------------------------

# Load cleaned dataset
file_path = r"C:\Users\Admin\Desktop\seasonal_sales_prediction\cleaned_data.xlsx"
df = pd.read_excel(file_path)

# Create Sales column
df['Sales'] = df['Quantity'] * df['UnitPrice']

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# -----------------------------
# Basic Statistical Measures
# -----------------------------
mean_sales = df['Sales'].mean()
median_sales = df['Sales'].median()
std_sales = df['Sales'].std()

print("Basic Sales Statistics")
print("----------------------")
print(f"Mean Sales   : {mean_sales:.2f}")
print(f"Median Sales : {median_sales:.2f}")
print(f"Std Deviation: {std_sales:.2f}")

print("\n-----------------------------------\n")

# -----------------------------
# Monthly Average Sales
# -----------------------------
df['Month'] = df['InvoiceDate'].dt.month
monthly_avg_sales = df.groupby('Month')['Sales'].mean()

print("Monthly Average Sales:")
print(monthly_avg_sales)

print("\n-----------------------------------\n")

# -----------------------------
# Yearly Sales Summary
# -----------------------------
df['Year'] = df['InvoiceDate'].dt.year
yearly_sales = df.groupby('Year')['Sales'].sum()

print("Yearly Sales Summary:")
print(yearly_sales)
