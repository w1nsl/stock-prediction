import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import sys

# Add the parent directory to the path so we can import from visualizations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualizations.predictions import (
    load_predictions,
    plot_prediction_comparison,
    plot_error_analysis,
    plot_error_distribution,
    calculate_metrics,
    plot_accuracy_vs_horizon,
    plot_performance_by_volatility,
    load_feature_importance,
    FEATURE_DESCRIPTIONS,
    MODEL_INTERPRETATIONS,
    load_model_evaluations,
    plot_model_comparison,
    plot_performance_over_time
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

# Get URL parameters
params = st.query_params

# Title and description
st.title("Stock Prediction Performance Dashboard")
st.markdown("""
This dashboard analyzes the performance of stock price predictions, showing comparison to actual prices,
error analysis, and model performance metrics.
""")

# Sidebar filters
st.sidebar.header("Filters")

# Stock selection with URL parameter support
default_stocks_str = params.get("stock_symbols", ["GOOG,AMD,COST,PYPL,QCOM,ABDE,PEP,CMCSA,INTC,SBUX"])[0]
default_stocks = [s.strip() for s in default_stocks_str.split(",")]

all_stocks = st.sidebar.text_input("Enter stock symbols (comma-separated)", ",".join(default_stocks))
all_stocks = [s.strip().upper() for s in all_stocks.split(",") if s.strip()]

# Default selected stock from URL params
default_selected_stock = params.get("selected_stock", [default_stocks[0] if default_stocks else "GOOG"])[0]
selected_stock = st.sidebar.selectbox("Select Stock", all_stocks, index=all_stocks.index(default_selected_stock) if default_selected_stock in all_stocks else 0)

# Date range selection with URL parameter support
today = datetime.now()
default_start = datetime.strptime("2023/06/16", "%Y/%m/%d").date()
default_end = datetime.strptime("2025/04/21", "%Y/%m/%d").date()

# Get date parameters from URL if provided
try:
    if "start_date" in params:
        default_start = datetime.strptime(params["start_date"][0], "%Y-%m-%d").date()
    if "end_date" in params:
        default_end = datetime.strptime(params["end_date"][0], "%Y-%m-%d").date()
except ValueError:
    # Fallback to defaults if dates are invalid
    pass

start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", default_end)

# Convert to string format for database query
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Update URL parameters
def update_url_params():
    st.query_params.update(
        stock_symbols=",".join(all_stocks),
        selected_stock=selected_stock,
        start_date=start_date_str,
        end_date=end_date_str
    )

# Uncomment to update URL when filters change - note this can cause refreshes
# update_url_params()

# Load data caching
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_cached_predictions(stock, start, end):
    try:
        # Try to load from database
        df = load_predictions(stock_symbol=stock, start_date=start, end_date=end)
        
        # If we got data back, return it
        if not df.empty:
            st.success(f"Successfully loaded prediction data from database")
            return df
            
        # If no data was found, return empty DataFrame
        st.error(f"No predictions found in database for {stock} between {start} and {end}")
        st.info("Please check your database connection or try a different stock/date range.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading predictions from database: {e}")
        st.info("Please check your database connection and credentials.")
        return pd.DataFrame()

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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Price Comparison", "Error Analysis", 
        "Error Distribution", "Advanced Analysis", 
        "Feature Importance", "Model Evaluation"
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
        
        # Create price bins - ensure at least 2 bins to avoid ValueError
        min_price = df['actual_price'].min()
        max_price = df['actual_price'].max()
        
        # Make sure we have enough range to create multiple bins
        if max_price - min_price < 0.01:
            # Almost no range, artificially create a range
            max_price = min_price + 10
        
        # Create 5 bins with proper number of labels (one fewer than bin edges)
        num_bins = 5
        bin_edges = np.linspace(min_price, max_price, num_bins + 1)
        bin_labels = [f"${bin_edges[i]:.0f}-${bin_edges[i+1]:.0f}" for i in range(num_bins)]
        
        # Create the price_bin column
        df['price_bin'] = pd.cut(
            df['actual_price'],
            bins=bin_edges,
            labels=bin_labels
        )
        
        # Calculate metrics by price bin - handle case with few or no bins
        if len(df['price_bin'].dropna().unique()) > 0:
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
        else:
            st.warning("Not enough price variation to create meaningful price bins.")
    
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
        
        # Load feature importance data with error handling
        try:
            with st.spinner("Loading feature importance data..."):
                feature_data = load_feature_importance()
                
            # Prepare data frames for both models
            rf_importance = None
            lgbm_importance = None
            
            # Handle both tuple format and dictionary format
            if isinstance(feature_data, tuple) and len(feature_data) == 2:
                # New format: tuple of DataFrames
                rf_importance, lgbm_importance = feature_data
            elif isinstance(feature_data, dict):
                # Old format: dictionary with model keys
                rf_importance = feature_data.get('random_forest')
                lgbm_importance = feature_data.get('lightgbm')
            else:
                st.warning("Feature importance data is in an unexpected format.")
            
            # Check if we have valid data for either model
            have_rf_data = rf_importance is not None and not (hasattr(rf_importance, 'empty') and rf_importance.empty)
            have_lgbm_data = lgbm_importance is not None and not (hasattr(lgbm_importance, 'empty') and lgbm_importance.empty)
            
            if have_rf_data or have_lgbm_data:
                # Create tabs for models that have data
                model_tabs_list = []
                if have_rf_data:
                    model_tabs_list.append("Random Forest")
                if have_lgbm_data:
                    model_tabs_list.append("LightGBM")
                
                model_tabs = st.tabs(model_tabs_list)
                
                # Display Random Forest data if available
                tab_index = 0
                if have_rf_data:
                    with model_tabs[tab_index]:
                        st.subheader("Random Forest Feature Importance")
                        st.write(MODEL_INTERPRETATIONS.get('random_forest', ''))
                        
                        # Sort and get top features
                        rf_top = rf_importance.sort_values('importance', ascending=False).head(10)
                        
                        # Create bar chart
                        fig = px.bar(rf_top, x='importance', y='feature', orientation='h',
                                    title='Top 10 Most Important Features - Random Forest',
                                    labels={'importance': 'Importance Score', 'feature': 'Feature'},
                                    color='importance', color_continuous_scale='Viridis')
                        
                        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display feature descriptions
                        st.subheader("Feature Descriptions")
                        feature_descriptions = []
                        for feature in rf_top['feature']:
                            description = FEATURE_DESCRIPTIONS.get(feature, "No description available")
                            feature_descriptions.append({
                                "Feature": feature,
                                "Description": description
                            })
                        
                        st.table(pd.DataFrame(feature_descriptions))
                    tab_index += 1
                
                # Display LightGBM data if available
                if have_lgbm_data:
                    with model_tabs[tab_index]:
                        st.subheader("LightGBM Feature Importance")
                        st.write(MODEL_INTERPRETATIONS.get('lightgbm', ''))
                        
                        # Sort and get top features
                        lgbm_top = lgbm_importance.sort_values('importance', ascending=False).head(10)
                        
                        # Create bar chart
                        fig = px.bar(lgbm_top, x='importance', y='feature', orientation='h',
                                    title='Top 10 Most Important Features - LightGBM',
                                    labels={'importance': 'Importance Score', 'feature': 'Feature'},
                                    color='importance', color_continuous_scale='Teal')
                        
                        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display feature descriptions
                        st.subheader("Feature Descriptions")
                        feature_descriptions = []
                        for feature in lgbm_top['feature']:
                            description = FEATURE_DESCRIPTIONS.get(feature, "No description available")
                            feature_descriptions.append({
                                "Feature": feature,
                                "Description": description
                            })
                        
                        st.table(pd.DataFrame(feature_descriptions))
            else:
                st.warning("No feature importance data is available for any model.")
                
        except Exception as e:
            st.error(f"Error loading feature importance data: {str(e)}")
            st.warning("Using the Feature Importance visualization requires data from trained models. Please check your database connection or model training status.")

    # Add the new tab for model evaluation
    with tab6:
        st.header("Model Evaluation Dashboard")
        st.markdown("""
        This tab provides insights into model performance across different stocks and over time.
        It helps identify which models are performing best and track improvements.
        """)
        
        # Load model evaluation data
        with st.spinner("Loading model evaluation data..."):
            model_eval_df = load_model_evaluations()
        
        if model_eval_df.empty:
            st.warning("No model evaluation data available in the database.")
        else:
            # Show summary metrics
            st.subheader("Model Performance Summary")
            
            # Display count of models by stock
            if 'stock_symbol' in model_eval_df.columns:
                stock_counts = model_eval_df['stock_symbol'].value_counts()
                st.write(f"Models for {len(stock_counts)} different stocks")
                
            # Display average metrics across all models    
            if 'rmse' in model_eval_df.columns:
                avg_rmse = model_eval_df['rmse'].mean()
                st.metric("Average RMSE", f"{avg_rmse:.4f}")
                
            if 'r2' in model_eval_df.columns:
                avg_r2 = model_eval_df['r2'].mean()
                st.metric("Average R²", f"{avg_r2:.4f}")
            
            # Create sections for different visualizations
            st.subheader("Model Comparison by Stock")
            model_comp_fig = plot_model_comparison(model_eval_df)
            if model_comp_fig:
                st.plotly_chart(model_comp_fig, use_container_width=True)
                st.info("""
                This chart shows model performance metrics across different stocks.
                Lower RMSE/MAE and higher R² indicate better performing models.
                """)
            else:
                st.warning("Could not create model comparison visualization.")
            
            # Only show time series if we have training_date
            if 'training_date' in model_eval_df.columns:
                st.subheader("Performance Trends Over Time")
                time_fig = plot_performance_over_time(model_eval_df)
                if time_fig:
                    st.plotly_chart(time_fig, use_container_width=True)
                    st.info("""
                    This chart shows how model performance has changed over time.
                    It helps identify trends and improvements in model accuracy.
                    """)
                else:
                    st.warning("Could not create performance over time visualization.")
            
            # Show raw data in expandable section
            with st.expander("View Raw Model Evaluation Data"):
                st.dataframe(model_eval_df)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
This dashboard analyzes prediction performance using various metrics and visualizations.
All data is loaded directly from the database with no sample data generation.
""")

# Run the app with: streamlit run visualizations/prediction_dashboard.py 