import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from visualizations import (
    load_data_from_db,
    plot_stock_candlestick,
    plot_sentiment_analysis,
    plot_correlation_heatmap,
    plot_economic_dashboard
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Stock Prediction & Analysis Dashboard")
st.markdown("""
This dashboard visualizes stock data, sentiment analysis, and economic indicators for your stock prediction project.
Select a stock and date range to explore the data and predictions.
""")

# Add demo mode notice
st.info("""
**Demo Mode**: This dashboard will display sample data if it cannot connect to the database. 
To use real data, ensure that your Airflow DAGs are running and populating the database tables.
""")

# Sidebar filters
st.sidebar.header("Filters")

# Stock selection
default_stocks = ["AAPL", "GOOG", "MSFT", "AMZN", "META"]
all_stocks = st.sidebar.text_input("Enter stock symbols (comma-separated)", ",".join(default_stocks))
all_stocks = [s.strip().upper() for s in all_stocks.split(",") if s.strip()]
selected_stock = st.sidebar.selectbox("Select Stock", all_stocks)

# Date range selection
today = datetime.now()
default_start = today - timedelta(days=90)  # 3 months for quicker loading of sample data
default_end = today

start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", default_end)

# Convert to string format for database query
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Load data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_cached_data(stock, start, end):
    try:
        return load_data_from_db(stock_symbol=stock, start_date=start, end_date=end)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

with st.spinner("Loading data..."):
    df = load_cached_data(selected_stock, start_date_str, end_date_str)

if df.empty:
    st.warning("No data found for the selected stock and date range. Please adjust your filters.")
else:
    # Display data source
    if 'data_source' in df.attrs and df.attrs['data_source'] == 'sample':
        st.warning("⚠️ Displaying sample data. Database connection failed or no data available.")
    
    # Main dashboard content
    st.sidebar.success(f"Loaded {len(df)} data points for {selected_stock}.")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Stock Price", "Sentiment Analysis", 
        "Economic Indicators", "Correlations"
    ])
    
    with tab1:
        st.header(f"{selected_stock} Stock Price")
        
        # Stock price chart
        candlestick_fig = plot_stock_candlestick(df, selected_stock)
        st.plotly_chart(candlestick_fig, use_container_width=True)
        
        # Key statistics
        st.subheader("Key Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate statistics
        if len(df) >= 2:
            latest_price = df['close_price'].iloc[-1]
            price_change = latest_price - df['close_price'].iloc[-2]
            percent_change = (price_change / df['close_price'].iloc[-2]) * 100
            
            highest_price = df['high_price'].max()
            lowest_price = df['low_price'].min()
            avg_volume = df['volume'].mean()
            
            col1.metric("Latest Price", f"${latest_price:.2f}", f"{percent_change:.2f}%")
            col2.metric("Highest Price", f"${highest_price:.2f}")
            col3.metric("Lowest Price", f"${lowest_price:.2f}")
            col4.metric("Avg Volume", f"{avg_volume:.0f}")
        else:
            st.warning("⚠️ Not enough data to compute key statistics. At least two data points are required.")
    
    with tab2:
        st.header(f"{selected_stock} Sentiment Analysis")
        
        # Sentiment chart
        sentiment_fig = plot_sentiment_analysis(df, selected_stock)
        st.plotly_chart(sentiment_fig, use_container_width=True)
        
        # Sentiment statistics
        st.subheader("Sentiment Statistics")
        col1, col2, col3 = st.columns(3)
        
        # Calculate statistics
        avg_sentiment = df['daily_sentiment'].mean()
        positive_days = (df['daily_sentiment'] > 0).sum() / len(df) * 100
        negative_days = (df['daily_sentiment'] < 0).sum() / len(df) * 100
        
        sentiment_color = "green" if avg_sentiment > 0 else "red"
        
        col1.metric("Average Sentiment", f"{avg_sentiment:.4f}")
        col2.metric("Positive Days", f"{positive_days:.1f}%")
        col3.metric("Negative Days", f"{negative_days:.1f}%")
        
        # Sentiment vs Price Change analysis
        st.subheader("Sentiment vs Next-Day Price Change")
        
        # Calculate next-day returns
        df_analysis = df.copy()
        df_analysis['next_day_close'] = df_analysis['close_price'].shift(-1)
        df_analysis['price_change_pct'] = (df_analysis['next_day_close'] - df_analysis['close_price']) / df_analysis['close_price'] * 100
        df_analysis = df_analysis.dropna()
        
        # Scatter plot of sentiment vs price change
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_analysis['daily_sentiment'],
                y=df_analysis['price_change_pct'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df_analysis['price_change_pct'],
                    colorscale='RdYlGn',
                    line=dict(width=1)
                ),
                text=df_analysis['date'].astype(str),
                hovertemplate='Date: %{text}<br>Sentiment: %{x:.4f}<br>Next-Day Change: %{y:.2f}%'
            )
        )
        
        fig.update_layout(
            title="Sentiment Score vs Next-Day Price Change",
            xaxis_title="Daily Sentiment Score",
            yaxis_title="Next-Day Price Change (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Economic Indicators")
        
        # Economic indicators dashboard
        econ_fig = plot_economic_dashboard(df, selected_stock)
        st.plotly_chart(econ_fig, use_container_width=True)
        
        # Economic indicators table
        st.subheader("Latest Economic Indicators")
        latest_date = df['date'].max()
        latest_econ = df[df['date'] == latest_date].iloc[0]
        
        econ_data = {
            "Indicator": ["S&P 500", "Fed Funds Rate", "Unemployment Rate", "CPI", "GDP", "Real GDP"],
            "Value": [
                latest_econ['sp500'],
                latest_econ['fed_funds_rate'],
                latest_econ['unemployment_rate'],
                latest_econ['cpi'],
                latest_econ['gdp'],
                latest_econ['real_gdp']
            ]
        }
        
        st.table(pd.DataFrame(econ_data))
    
    with tab4:
        st.header("Correlation Analysis")
        
        # Display correlation heatmap
        corr_plt = plot_correlation_heatmap(df, selected_stock)
        st.plotly_chart(corr_plt, use_container_width=True)
        
        # Feature selection for detailed correlation
        st.subheader("Detailed Correlation Analysis")
        
        # Available features
        features = [
            'close_price', 'daily_sentiment', 'article_count',
            'positive_ratio', 'negative_ratio', 'gdp', 'real_gdp', 
            'unemployment_rate', 'cpi', 'fed_funds_rate', 'sp500'
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature_x = st.selectbox("X-axis Feature", features, index=0)
        
        with col2:
            feature_y = st.selectbox("Y-axis Feature", features, index=1)
            
        # Scatter plot of selected features
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df[feature_x],
                y=df[feature_y],
                mode='markers',
                marker=dict(
                    size=8,
                    color=[pd.Timestamp(d).timestamp() for d in df['date']],
                    colorscale='Viridis',
                    line=dict(width=1),
                    showscale=True,
                    colorbar=dict(title="Date")
                ),
                text=df['date'].astype(str),
                hovertemplate='Date: %{text}<br>' + feature_x + ': %{x}<br>' + feature_y + ': %{y}'
            )
        )
        
        # Calculate and display correlation coefficient
        correlation = df[feature_x].corr(df[feature_y])
        
        fig.update_layout(
            title=f"Correlation between {feature_x} and {feature_y}: {correlation:.4f}",
            xaxis_title=feature_x,
            yaxis_title=feature_y,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard uses data from your stock prediction project, "
    "combining stock prices, sentiment analysis, and economic indicators."
)

# Run the app with: streamlit run visualizations/dashboard.py 