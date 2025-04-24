import streamlit as st
import pandas as pd
import numpy as np

# Basic page config
st.set_page_config(
    page_title="Streamlit Native Visualizations",
    page_icon="📊",
    layout="wide"
)

st.title("Stock Dashboard - Streamlit Native Visualizations")
st.write("Using Streamlit's built-in visualization capabilities")

# Create some sample data
def generate_stock_data():
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    price = 100 + np.cumsum(np.random.normal(0, 1, 100))
    volume = np.random.randint(1000, 10000, 100)
    return pd.DataFrame({
        'Date': dates,
        'Price': price,
        'Volume': volume,
        'MA7': pd.Series(price).rolling(7).mean(),
        'Return': pd.Series(price).pct_change() * 100
    })

# Generate sample data
data = generate_stock_data()

# Sidebar filters
st.sidebar.title("Dashboard Controls")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(data['Date'].min().date(), data['Date'].max().date())
)

# Convert numpy data types to native Python types for JSON serialization
data_clean = data.copy()
for col in data_clean.columns:
    if pd.api.types.is_numeric_dtype(data_clean[col]):
        data_clean[col] = data_clean[col].astype(float)

# Filter data based on date selection
mask = (data_clean['Date'] >= pd.Timestamp(date_range[0])) & (data_clean['Date'] <= pd.Timestamp(date_range[1]))
filtered_data = data_clean[mask]

# Main dashboard content
st.header("Stock Price Analysis")

# Price chart using Streamlit's native line_chart
st.subheader("Price Chart")
st.line_chart(filtered_data.set_index('Date')['Price'])

# Moving average comparison
st.subheader("Price vs 7-Day Moving Average")
ma_chart_data = filtered_data.set_index('Date')[['Price', 'MA7']]
st.line_chart(ma_chart_data)

# Volume chart
st.subheader("Trading Volume")
st.bar_chart(filtered_data.set_index('Date')['Volume'])

# Return distribution
st.subheader("Daily Return Distribution")
st.bar_chart(filtered_data.set_index('Date')['Return'])

# Raw data display
st.subheader("Raw Data")
st.dataframe(filtered_data)

# Statistics
st.header("Key Statistics")
col1, col2, col3 = st.columns(3)

# Calculate key metrics
avg_price = filtered_data['Price'].mean()
max_price = filtered_data['Price'].max()
min_price = filtered_data['Price'].min()
volatility = filtered_data['Return'].std()

col1.metric("Average Price", f"${avg_price:.2f}")
col2.metric("Max Price", f"${max_price:.2f}")
col3.metric("Min Price", f"${min_price:.2f}")

col1.metric("Volatility", f"{volatility:.2f}%")
col2.metric("Price Change", f"{(filtered_data['Price'].iloc[-1] - filtered_data['Price'].iloc[0]):.2f}")
col3.metric("% Change", f"{((filtered_data['Price'].iloc[-1] / filtered_data['Price'].iloc[0]) - 1) * 100:.2f}%") 