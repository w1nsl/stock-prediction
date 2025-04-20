import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from visualizations import (
    load_predictions,
    plot_prediction_comparison,
    plot_error_analysis,
    plot_error_distribution,
    calculate_metrics,
    plot_accuracy_vs_horizon,
    plot_performance_by_volatility,
    plot_feature_importance,
    load_feature_importance
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Stock Prediction Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Stock Prediction Performance Dashboard")
st.markdown("""
This dashboard analyzes the performance of stock price predictions, showing comparison to actual prices,
error analysis, and model performance metrics.
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
default_start = today - timedelta(days=365)  # 1 year ago
default_end = today

start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", default_end)

# Convert to string format for database query
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Load data caching
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_cached_predictions(stock, start, end):
    try:
        # Try to load from database (this will fail if table doesn't exist)
        return load_predictions(stock_symbol=stock, start_date=start, end_date=end)
    except Exception as e:
        st.warning(f"Error loading predictions from database: {e}")
        st.info("Generating sample prediction data for demonstration")
        
        # Generate sample data for demonstration
        dates = pd.date_range(start=start, end=end)
        
        # Create dummy prediction data
        np.random.seed(42)  # For reproducibility
        base_price = 150 if stock == "AAPL" else 100
        volatility = 5 if stock == "AAPL" else 3
        
        actual_prices = np.linspace(base_price, base_price * 1.3, len(dates)) + np.random.normal(0, volatility, len(dates))
        predicted_prices = actual_prices + np.random.normal(0, volatility * 2, len(dates))
        
        # Create DataFrame
        return pd.DataFrame({
            'date': dates,
            'stock_symbol': stock,
            'actual_price': actual_prices,
            'predicted_price': predicted_prices
        })

with st.spinner("Loading prediction data..."):
    df = load_cached_predictions(selected_stock, start_date_str, end_date_str)

if df.empty:
    st.warning("No prediction data found for the selected stock and date range. Please adjust your filters.")
else:
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Main dashboard content
    st.sidebar.success(f"Loaded {len(df)} prediction data points for {selected_stock}")
    
    # Metrics section
    st.subheader("Prediction Performance Metrics")
    cols = st.columns(5)
    
    cols[0].metric("RMSE", f"{metrics['RMSE']:.4f}")
    cols[1].metric("MAE", f"{metrics['MAE']:.4f}")
    cols[2].metric("MAPE", f"{metrics['MAPE']:.2f}%")
    cols[3].metric("R²", f"{metrics['R²']:.4f}")
    cols[4].metric("Direction Accuracy", f"{metrics['Direction Accuracy']:.2f}%")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Price Comparison", "Error Analysis", 
        "Error Distribution", "Advanced Analysis", "Feature Importance"
    ])
    
    with tab1:
        st.header(f"{selected_stock} - Actual vs. Predicted Prices")
        
        # Price comparison chart
        comparison_fig = plot_prediction_comparison(df, selected_stock)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Period analysis
        st.subheader("Performance by Time Period")
        
        # Add period selector
        period_options = ["Full Period", "Last Month", "Last Quarter", "Last 6 Months"]
        selected_period = st.selectbox("Select Period", period_options)
        
        # Filter data based on period
        period_df = df.copy()
        if selected_period == "Last Month":
            cutoff_date = end_date - timedelta(days=30)
            period_df = df[df['date'] >= cutoff_date]
        elif selected_period == "Last Quarter":
            cutoff_date = end_date - timedelta(days=90)
            period_df = df[df['date'] >= cutoff_date]
        elif selected_period == "Last 6 Months":
            cutoff_date = end_date - timedelta(days=180)
            period_df = df[df['date'] >= cutoff_date]
        
        if len(period_df) > 0:
            period_metrics = calculate_metrics(period_df)
            
            # Display period metrics
            period_cols = st.columns(5)
            period_cols[0].metric("Period RMSE", f"{period_metrics['RMSE']:.4f}")
            period_cols[1].metric("Period MAE", f"{period_metrics['MAE']:.4f}")
            period_cols[2].metric("Period MAPE", f"{period_metrics['MAPE']:.2f}%")
            period_cols[3].metric("Period R²", f"{period_metrics['R²']:.4f}")
            period_cols[4].metric("Period Direction Accuracy", f"{period_metrics['Direction Accuracy']:.2f}%")
        else:
            st.warning(f"No data available for the selected period: {selected_period}")
    
    with tab2:
        st.header(f"{selected_stock} - Prediction Error Analysis")
        
        # Error analysis chart
        error_fig = plot_error_analysis(df, selected_stock)
        st.plotly_chart(error_fig, use_container_width=True)
        
        # Error statistics
        st.subheader("Error Statistics")
        
        # Calculate basic error statistics
        df['error'] = df['actual_price'] - df['predicted_price']
        df['abs_error'] = abs(df['error'])
        df['pct_error'] = (df['error'] / df['actual_price']) * 100
        
        error_stats = {
            "Metric": ["Mean Error", "Mean Absolute Error", "Mean % Error", "Max Overestimation", "Max Underestimation"],
            "Value": [
                f"{df['error'].mean():.4f}",
                f"{df['abs_error'].mean():.4f}",
                f"{df['pct_error'].mean():.2f}%",
                f"{df['error'].min():.4f}",
                f"{df['error'].max():.4f}"
            ]
        }
        
        st.table(pd.DataFrame(error_stats))
        
        # Show days with largest errors
        st.subheader("Days with Largest Prediction Errors")
        
        largest_errors = df.sort_values('abs_error', ascending=False).head(5)
        largest_errors = largest_errors[['date', 'actual_price', 'predicted_price', 'error', 'pct_error']]
        largest_errors = largest_errors.rename(columns={
            'date': 'Date',
            'actual_price': 'Actual Price',
            'predicted_price': 'Predicted Price',
            'error': 'Error',
            'pct_error': '% Error'
        })
        
        st.dataframe(largest_errors.style.format({
            'Actual Price': '${:.2f}',
            'Predicted Price': '${:.2f}',
            'Error': '${:.2f}',
            '% Error': '{:.2f}%'
        }), use_container_width=True)
    
    with tab3:
        st.header(f"{selected_stock} - Error Distribution Analysis")
        
        # Error distribution chart
        distribution_fig = plot_error_distribution(df, selected_stock)
        st.plotly_chart(distribution_fig, use_container_width=True)
        
        # Error percentiles
        st.subheader("Error Percentiles")
        
        percentiles = [10, 25, 50, 75, 90]
        error_percentiles = np.percentile(df['error'], percentiles)
        abs_error_percentiles = np.percentile(df['abs_error'], percentiles)
        pct_error_percentiles = np.percentile(df['pct_error'], percentiles)
        
        percentile_data = {
            "Percentile": [f"{p}%" for p in percentiles],
            "Error": [f"${e:.2f}" for e in error_percentiles],
            "Absolute Error": [f"${e:.2f}" for e in abs_error_percentiles],
            "Percentage Error": [f"{e:.2f}%" for e in pct_error_percentiles]
        }
        
        st.table(pd.DataFrame(percentile_data))
        
        # Error distribution by price range
        st.subheader("Error by Price Range")
        
        # Create price bins
        min_price = df['actual_price'].min()
        max_price = df['actual_price'].max()
        step = (max_price - min_price) / 5
        
        df['price_bin'] = pd.cut(
            df['actual_price'],
            bins=np.arange(min_price, max_price + step, step),
            labels=[f"${min_price + i*step:.0f}-${min_price + (i+1)*step:.0f}" for i in range(5)]
        )
        
        # Calculate metrics by price bin
        bin_metrics = df.groupby('price_bin').agg(
            mean_error=('error', 'mean'),
            mean_abs_error=('abs_error', 'mean'),
            mean_pct_error=('pct_error', 'mean'),
            count=('error', 'count')
        ).reset_index()
        
        # Create horizontal bar chart for error by price range
        fig = go.Figure()
        
        # Add bars for each metric
        fig.add_trace(go.Bar(
            y=bin_metrics['price_bin'],
            x=bin_metrics['mean_abs_error'],
            name='Mean Absolute Error',
            orientation='h',
            marker_color='orange'
        ))
        
        fig.add_trace(go.Bar(
            y=bin_metrics['price_bin'],
            x=bin_metrics['mean_pct_error'],
            name='Mean Percentage Error',
            orientation='h',
            marker_color='green',
            visible='legendonly'  # Hide by default
        ))
        
        fig.update_layout(
            title="Prediction Error by Price Range",
            xaxis_title="Error Value",
            yaxis_title="Price Range",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Advanced Performance Analysis")
        
        # Analysis selection
        analysis_type = st.radio(
            "Select Analysis Type",
            ["Accuracy vs Prediction Horizon", "Performance vs Volatility"],
            horizontal=True
        )
        
        if analysis_type == "Accuracy vs Prediction Horizon":
            st.subheader("Prediction Accuracy vs Horizon")
            
            # Configure horizon
            max_horizon = st.slider("Maximum Prediction Horizon (Days)", 5, 30, 10)
            
            # Generate horizon analysis
            horizon_fig = plot_accuracy_vs_horizon(df, selected_stock, max_days=max_horizon)
            st.plotly_chart(horizon_fig, use_container_width=True)
            
            st.info("""
            This analysis shows how the model performs when making predictions for different time horizons. 
            Lower RMSE/MAE/MAPE and higher Direction Accuracy indicate better performance.
            """)
            
        else:  # Performance vs Volatility
            st.subheader("Prediction Performance vs Market Volatility")
            
            # Configure volatility window
            volatility_window = st.slider("Volatility Window (Days)", 10, 60, 20)
            
            # Generate volatility analysis
            volatility_fig = plot_performance_by_volatility(df, selected_stock, window=volatility_window)
            st.plotly_chart(volatility_fig, use_container_width=True)
            
            st.info("""
            This analysis shows the relationship between market volatility and prediction error.
            A strong positive correlation indicates that the model struggles more during volatile periods.
            """)
    
    with tab5:
        st.header("Model Feature Importance Analysis")
        st.markdown("""
        This tab shows which features have the most influence on the prediction models.
        Understanding feature importance helps identify the key drivers of stock price movements.
        """)
        
        # Load feature importance data
        feature_importance_data = load_feature_importance()
        
        # Model selector
        model_options = ["Random Forest", "LightGBM"]
        selected_model = st.selectbox("Select Model", model_options)
        
        # Map selection to data key
        model_key = "random_forest" if selected_model == "Random Forest" else "lightgbm"
        
        # Get data for selected model
        model_data = feature_importance_data[model_key]
        
        if not model_data.empty:
            # Create visualization
            st.subheader(f"{selected_model} Feature Importance")
            
            # Plot top N features
            top_n = st.slider("Number of features to display", 5, 25, 10)
            top_features = model_data.head(top_n)
            
            # Create visualization
            fig = plot_feature_importance(
                feature_importance=top_features['importance'].values,
                feature_names=top_features['feature'].values
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature descriptions for better understanding
            st.subheader("Feature Descriptions")
            
            feature_descriptions = {
                'volume': 'Trading volume (number of shares traded)',
                'daily_sentiment': 'Overall sentiment score from news articles',
                'article_count': 'Number of news articles published about the stock',
                'positive_ratio': 'Percentage of positive sentiment in news articles',
                'negative_ratio': 'Percentage of negative sentiment in news articles',
                'neutral_ratio': 'Percentage of neutral sentiment in news articles',
                'real_gdp': 'Real Gross Domestic Product (economic indicator)',
                'unemployment_rate': 'Current unemployment rate percentage',
                'cpi': 'Consumer Price Index (inflation measure)',
                'fed_funds_rate': 'Federal Reserve interest rate',
                'return_1d': 'Previous 1-day price return',
                'return_3d': 'Previous 3-day price return',
                'return_5d': 'Previous 5-day price return',
                'ma7': '7-day moving average price',
                'rsi': 'Relative Strength Index (momentum indicator)',
                'volatility_7d': '7-day price volatility',
                'volume_ma5': '5-day moving average of trading volume',
                'volume_change': 'Daily change in trading volume',
                'sentiment_volume': 'Sentiment weighted by article volume',
                'sentiment_ma3': '3-day moving average of sentiment',
                'high_news_day': 'Flag for days with high news coverage',
                'fed_rate_increase': 'Flag for Federal Reserve rate increase',
                'day_sin': 'Cyclical encoding of day (sine component)',
                'day_cos': 'Cyclical encoding of day (cosine component)',
                'month_end': 'Flag for end of month'
            }
            
            # Create a filtered description table for the top features
            top_descriptions = {
                'Feature': list(top_features['feature']),
                'Description': [feature_descriptions.get(feat, "No description available") 
                               for feat in top_features['feature']]
            }
            
            st.table(pd.DataFrame(top_descriptions))
            
            # Additional interpretation
            st.subheader("Model Interpretation")
            
            # Different interpretation based on model
            if model_key == "random_forest":
                st.write("""
                The Random Forest model calculates feature importance based on how much each feature 
                reduces impurity when used in decision trees. Features with higher importance 
                contribute more to the prediction outcome and represent stronger predictors of 
                price movement.
                """)
            else:  # LightGBM
                st.write("""
                The LightGBM model calculates feature importance based on the gain achieved when a 
                feature is used for splitting. Features with higher importance contribute more to 
                prediction accuracy and represent the most influential factors in predicting 
                stock price movements.
                """)
        else:
            st.warning("No feature importance data available for the selected model.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
This dashboard analyzes prediction performance using various metrics and visualizations.
For demonstration purposes, sample data is generated if no prediction data is available in the database.
""")

# Run the app with: streamlit run visualizations/prediction_dashboard.py 