import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Module 3: Exploratory Data Analysis
# -----------------------------

# Load cleaned dataset
file_path = r"C:\Users\Admin\Desktop\seasonal_sales_prediction\cleaned_data.xlsx"
df = pd.read_excel(file_path)

# Create Sales column
df['Sales'] = df['Quantity'] * df['UnitPrice']

# Convert InvoiceDate to datetime (safety)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Extract Month and Year
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month

# -----------------------------
# Monthly Sales Analysis
# -----------------------------
monthly_sales = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()

print("Monthly Sales Data:")
print(monthly_sales.head())

# -----------------------------
# Plot Monthly Sales Trend
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(monthly_sales['Month'], monthly_sales['Sales'], marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()
