import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Seasonal Sales Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Seasonal Sales Prediction Dashboard")

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    file_path = r"C:\Users\Admin\Desktop\seasonal_sales_prediction\Online Retail-Copy1.xlsx"
    df = pd.read_excel(file_path)
    return df

df = load_data()

# =====================================
# DATA PREPARATION
# =====================================
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Sales'] = df['Quantity'] * df['UnitPrice']

# Remove year 2010
df = df[df['Year'] != 2010]

# =====================================
# SIDEBAR FILTERS (INNOVATION 1)
# =====================================
st.sidebar.header("🔎 Filter Options")

selected_year = st.sidebar.selectbox(
    "Select Year",
    sorted(df['Year'].unique())
)

filtered_df = df[df['Year'] == selected_year]

monthly_sales = (
    filtered_df.groupby('Month')['Sales']
    .sum()
    .reset_index()
)

# =====================================
# KPI SECTION
# =====================================
st.subheader("📌 Key Performance Indicators")

total_sales = filtered_df['Sales'].sum()
total_orders = filtered_df['InvoiceNo'].nunique()
total_customers = filtered_df['CustomerID'].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Sales", f"{total_sales:,.2f}")
col2.metric("🧾 Total Orders", total_orders)
col3.metric("👥 Total Customers", total_customers)

st.divider()

# =====================================
# SALES TREND CHART
# =====================================
st.subheader("📈 Monthly Sales Trend")

fig_sales = px.line(
    monthly_sales,
    x="Month",
    y="Sales",
    markers=True,
    title="Monthly Sales Trend"
)

st.plotly_chart(fig_sales, width="stretch")

# =====================================
# 🚀 INNOVATION 4: MODEL ACCURACY METRICS
# =====================================
st.subheader("📐 Model Performance Metrics")

X = monthly_sales[['Month']]
y = monthly_sales['Sales']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

m1, m2, m3 = st.columns(3)

m1.metric("📊 R² Score", f"{r2:.3f}")
m2.metric("📉 MAE", f"{mae:,.2f}")
m3.metric("📈 RMSE", f"{rmse:,.2f}")

st.divider()


# =====================================
# 🚀 INNOVATION 2: BEST & WORST MONTH
# =====================================
st.subheader("📅 Best & Worst Performing Months")

best_month = monthly_sales.loc[monthly_sales['Sales'].idxmax()]
worst_month = monthly_sales.loc[monthly_sales['Sales'].idxmin()]

b1, b2 = st.columns(2)

b1.success(
    f"🔥 Best Month: {int(best_month['Month'])}\n\n"
    f"Sales: {best_month['Sales']:,.2f}"
)

b2.error(
    f"📉 Worst Month: {int(worst_month['Month'])}\n\n"
    f"Sales: {worst_month['Sales']:,.2f}"
)

st.divider()

# =====================================
# DATA TABLE
# =====================================
st.subheader("📋 Monthly Sales Data")

st.dataframe(monthly_sales, width="stretch")

# =====================================
# 🚀 INNOVATION 3: DOWNLOAD REPORT
# =====================================
st.subheader("📥 Download Reports")

csv_data = monthly_sales.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Monthly Sales Report (CSV)",
    data=csv_data,
    file_name=f"monthly_sales_{selected_year}.csv",
    mime="text/csv"
)

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.caption("Seasonal Sales Prediction | ML Dashboard Project")
# =====================================
# 🚀 INNOVATION 5: AUTOMATIC TEXT INSIGHTS
# =====================================
st.subheader("🧠 Automated Insights")

trend = (
    "increasing"
    if monthly_sales['Sales'].iloc[-1] > monthly_sales['Sales'].iloc[0]
    else "decreasing"
)

insight_text = f"""
### 📊 Sales Trend Analysis
- Sales show an overall **{trend} trend** during {selected_year}.
- The **best performing month** is Month **{int(best_month['Month'])}**, indicating peak demand.
- The **lowest sales** occurred in Month **{int(worst_month['Month'])}**, suggesting off-season impact.

### 🤖 Model Performance
- The model achieved an **R² score of {r2:.2f}**, showing a good fit.
- The average prediction error (MAE) is **{mae:,.2f}**.

### 📌 Business Recommendation
- Increase stock before peak months.
- Run promotions during low-performing months.
"""

st.markdown(insight_text)

# =====================================
# 🚀 INNOVATION 6: FUTURE SALES FORECAST
# =====================================
st.subheader("🔮 Future Sales Forecast")

# Select forecast horizon
forecast_months = st.slider(
    "Select number of months to forecast",
    min_value=3,
    max_value=12,
    value=6
)

# Generate future months
last_month = monthly_sales['Month'].max()
future_months = np.arange(last_month + 1, last_month + forecast_months + 1)

# Wrap months after December
future_months = [(m - 1) % 12 + 1 for m in future_months]

future_df = pd.DataFrame({'Month': future_months})

# Predict future sales
future_df['Predicted_Sales'] = model.predict(future_df[['Month']])

# Plot actual vs forecast
forecast_fig = px.line(
    title="📈 Actual vs Forecasted Sales",
)

forecast_fig.add_scatter(
    x=monthly_sales['Month'],
    y=monthly_sales['Sales'],
    mode='lines+markers',
    name='Actual Sales'
)

forecast_fig.add_scatter(
    x=future_df['Month'],
    y=future_df['Predicted_Sales'],
    mode='lines+markers',
    name='Forecasted Sales',
    line=dict(dash='dash')
)

forecast_fig.update_layout(
    xaxis_title="Month",
    yaxis_title="Sales"
)

st.plotly_chart(forecast_fig, width="stretch")

# Display forecast table
st.subheader("📋 Forecasted Sales Table")
st.dataframe(future_df, width="stretch")
# =====================================
# 🚀 INNOVATION 7: ANOMALY / OUTLIER DETECTION
# =====================================
st.subheader("🚨 Sales Anomaly Detection")

# Calculate Z-score
sales_mean = monthly_sales['Sales'].mean()
sales_std = monthly_sales['Sales'].std()

monthly_sales['Z_score'] = (
    (monthly_sales['Sales'] - sales_mean) / sales_std
)

# Detect anomalies
anomalies = monthly_sales[monthly_sales['Z_score'].abs() > 2]

if anomalies.empty:
    st.success("✅ No significant sales anomalies detected.")
else:
    st.warning("⚠️ Sales anomalies detected!")

    st.dataframe(
        anomalies[['Month', 'Sales', 'Z_score']],
        width="stretch"
    )

    # Highlight anomalies in chart
    anomaly_fig = px.scatter(
        monthly_sales,
        x="Month",
        y="Sales",
        title="🚨 Detected Sales Anomalies",
        color=monthly_sales['Z_score'].abs() > 2,
        color_discrete_map={True: "red", False: "blue"}
    )

    st.plotly_chart(anomaly_fig, width="stretch")
# =====================================
# 🚀 INNOVATION 8: CUSTOMER SEGMENTATION
# =====================================
st.subheader("👥 Customer Segmentation (Clustering)")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Prepare customer-level data
customer_df = (
    filtered_df
    .groupby('CustomerID')
    .agg(
        Total_Sales=('Sales', 'sum'),
        Total_Orders=('InvoiceNo', 'nunique')
    )
    .reset_index()
    .dropna()
)

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(
    customer_df[['Total_Sales', 'Total_Orders']]
)

# Select number of clusters
k = st.slider(
    "Select number of customer segments (clusters)",
    min_value=2,
    max_value=5,
    value=3
)

# Apply KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
customer_df['Cluster'] = kmeans.fit_predict(scaled_features)

# Cluster visualization
cluster_fig = px.scatter(
    customer_df,
    x="Total_Orders",
    y="Total_Sales",
    color="Cluster",
    title="👥 Customer Segments based on Buying Behavior",
    labels={
        "Total_Orders": "Number of Orders",
        "Total_Sales": "Total Sales"
    }
)

st.plotly_chart(cluster_fig, width="stretch")

# Show clustered data
st.subheader("📋 Customer Segmentation Table")
st.dataframe(customer_df.head(20), width="stretch")
# =====================================
# 🚀 INNOVATION 9: SMART BUSINESS RECOMMENDATIONS
# =====================================
st.subheader("💡 Smart Business Recommendations")

recommendations = []

# Trend-based recommendation
if trend == "increasing":
    recommendations.append(
        "📈 Sales are increasing. Consider increasing inventory and expanding marketing efforts."
    )
else:
    recommendations.append(
        "📉 Sales are decreasing. Consider offering discounts or revising pricing strategies."
    )

# Best & worst month recommendations
recommendations.append(
    f"🔥 Focus promotions before Month {int(best_month['Month'])} to maximize peak sales."
)

recommendations.append(
    f"📉 Introduce special offers during Month {int(worst_month['Month'])} to improve low sales."
)

# Model reliability recommendation
if r2 > 0.7:
    recommendations.append(
        "🤖 The prediction model is reliable. Forecasted sales can be used for planning."
    )
else:
    recommendations.append(
        "⚠️ The prediction model has moderate accuracy. Use forecasts cautiously."
    )

# Display recommendations
for rec in recommendations:
    st.success(rec)
# ==============================
# INNOVATION 10: SMART INSIGHTS
# ==============================

st.subheader("📌 Smart Business Insights")

# Best & Worst Month
best_month = monthly_sales.loc[monthly_sales['Sales'].idxmax()]
worst_month = monthly_sales.loc[monthly_sales['Sales'].idxmin()]

# Sales Trend
sales_diff = monthly_sales['Sales'].diff().mean()

if sales_diff > 0:
    trend = "📈 Upward Growth"
    trend_msg = "Sales are generally increasing over time."
else:
    trend = "📉 Downward Decline"
    trend_msg = "Sales show a declining trend."

# Display insights
st.markdown(f"""
### 🔍 Key Insights
- 🏆 **Best Month:** Month **{int(best_month['Month'])}**
- ⚠️ **Worst Month:** Month **{int(worst_month['Month'])}**
- 📊 **Sales Trend:** {trend}

### 🧠 Interpretation
- The highest sales occur during **Month {int(best_month['Month'])}**, suggesting peak demand.
- Sales drop significantly in **Month {int(worst_month['Month'])}**, indicating an opportunity for promotions.
- {trend_msg}

### ✅ Recommendations
- 📦 Increase inventory before **Month {int(best_month['Month'])}**
- 💸 Offer discounts during **Month {int(worst_month['Month'])}**
- 📣 Launch marketing campaigns during declining months
""")
# ==============================
# INNOVATION 11: SALES FORECASTING
# ==============================

st.subheader("🔮 Sales Forecasting (Future Prediction)")

# Prepare data for ML
X = monthly_sales[['Month']]
y = monthly_sales['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next 3 months
last_month = int(monthly_sales['Month'].max())
future_months = np.array([[last_month + 1],
                          [last_month + 2],
                          [last_month + 3]])

future_sales = model.predict(future_months)

# Create dataframe for forecast
forecast_df = pd.DataFrame({
    'Month': future_months.flatten(),
    'Predicted Sales': future_sales
})

# Display forecast table
st.write("📊 Predicted Sales for Upcoming Months")
st.dataframe(forecast_df)

# Plot forecast
fig_forecast = px.line(
    pd.concat([
        monthly_sales.rename(columns={'Sales': 'Value'}).assign(Type='Actual'),
        forecast_df.rename(columns={'Predicted Sales': 'Value'}).assign(Type='Forecast')
    ]),
    x='Month',
    y='Value',
    color='Type',
    markers=True,
    title="Actual vs Forecasted Sales"
)

st.plotly_chart(fig_forecast, width="stretch")

