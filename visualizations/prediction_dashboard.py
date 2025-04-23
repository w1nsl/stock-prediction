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

# Add debug mode toggle
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# Stock selection with URL parameter support
default_stocks_str = params.get("stock_symbols", ["GOOG,AMD,COST,PYPL,QCOM,ADBE,PEP,CMCSA,INTC,SBUX"])[0]
default_stocks = [s.strip() for s in default_stocks_str.split(",")]

# Ensure single character stocks are properly expanded
# This fixes the issue where "G" is shown instead of "GOOG"
default_stocks = [stock if len(stock) > 1 else "GOOG" if stock == "G" else stock for stock in default_stocks]

all_stocks = st.sidebar.text_input("Enter stock symbols (comma-separated)", ",".join(default_stocks))
all_stocks = [s.strip().upper() for s in all_stocks.split(",") if s.strip()]

# Ensure the selected stock exists in the list and is properly expanded
if len(all_stocks) == 1 and len(all_stocks[0]) == 1 and all_stocks[0] == "G":
    all_stocks = ["GOOG"]

# Default selected stock from URL params
default_selected_stock = params.get("selected_stock", [default_stocks[0] if default_stocks else "GOOG"])[0]
selected_stock = st.sidebar.selectbox("Select Stock", all_stocks, index=all_stocks.index(default_selected_stock) if default_selected_stock in all_stocks else 0)

# Date range selection with URL parameter support
today = datetime.now()
default_start = datetime.strptime("2023/06/16", "%Y/%m/%d").date()
default_end = datetime.strptime("2023/12/31", "%Y/%m/%d").date()

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Price Comparison", "Error Analysis", 
        "Error Distribution", "Advanced Analysis", 
        "Model Evaluation"
    ])
    
    with tab1:
        st.header(f"{selected_stock} - Actual vs. Predicted Prices")
        
        # Add description for chart interpretation
        st.markdown("""
        **How to interpret this chart:**
        - **Blue line**: Historical actual stock prices
        - **Red line**: Historical prediction line
        - **Yellow line**: Latest prediction segment (connecting the last two predictions)
        - **Green dashed line**: Rolling average of actual prices
        
        **Interactions available:**
        - Hover over data points to see exact values
        - Use the toolbar on the right to zoom, pan, or download the chart
        - Double-click to reset the view
        """)
        
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
        
        # Make sure the date column is in datetime format for comparison
        if not pd.api.types.is_datetime64_any_dtype(period_df['date']):
            period_df['date'] = pd.to_datetime(period_df['date'])
        
        # Convert end_date to datetime for comparison
        end_datetime = pd.to_datetime(end_date)
        
        if selected_period == "Last Month":
            cutoff_date = end_datetime - timedelta(days=30)
            period_df = period_df[period_df['date'] >= cutoff_date]
        elif selected_period == "Last Quarter":
            cutoff_date = end_datetime - timedelta(days=90)
            period_df = period_df[period_df['date'] >= cutoff_date]
        elif selected_period == "Last 6 Months":
            cutoff_date = end_datetime - timedelta(days=180)
            period_df = period_df[period_df['date'] >= cutoff_date]
        
        # Debug information
        st.caption(f"Period: {selected_period}, Data Points: {len(period_df)}")
        if selected_period != "Full Period":
            st.caption(f"Cutoff Date: {cutoff_date.strftime('%Y-%m-%d')}, End Date: {end_datetime.strftime('%Y-%m-%d')}")
        
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
        
        # Add description for chart interpretation
        st.markdown("""
        **How to interpret these charts:**
        - **Top panel (Raw Error)**: Shows the difference between actual and predicted prices over time
          - Values above zero: Underestimated predictions (actual price was higher)
          - Values below zero: Overestimated predictions (predicted price was higher)
          - Red dotted line: Mean error
        
        - **Middle panel (Absolute Error)**: Shows the magnitude of errors regardless of direction
          - Higher values indicate larger prediction errors
          - Filled area highlights error magnitude
        
        - **Bottom panel (Percentage Error)**: Shows error relative to actual price
          - Normalizes errors across different price levels
          - Useful for comparing accuracy across time periods with different price ranges
        
        **What to look for:**
        - Patterns in errors over time (consistently over/under-predicting)
        - Time periods with unusually high errors
        - Correlation between errors and market events
        """)
        
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
        
        # Add description for chart interpretation
        st.markdown("""
        **How to interpret these histograms:**
        - **Left panel (Error Distribution)**: Shows the distribution of raw prediction errors
          - Center (0): Perfect predictions
          - Right side (>0): Underestimations (predicted too low)
          - Left side (<0): Overestimations (predicted too high)
          - Bell-shaped/normal distribution: Errors are random and unbiased
          - Skewed distribution: Systematic bias in predictions
        
        - **Right panel (Percentage Error)**: Shows the distribution of percent errors
          - Helps understand relative impact of errors
          - Wide distribution: High prediction variability
          - Narrow distribution: Consistent prediction performance
        
        **What to look for:**
        - Symmetry around zero (balanced predictions)
        - Outliers or extreme values (unusual prediction errors)
        - Multiple peaks (different prediction regimes or market conditions)
        """)
        
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
            This analysis shows the relationship between market volatility and prediction error. A strong positive correlation indicates that the model struggles more during volatile periods.
            """)
            
            # Add more detailed interpretation guidance
            st.markdown("""
            **How to interpret this graph:**
            
            - **Scatter Points**: Each point represents a prediction, colored by date (newer predictions are lighter).
            - **Red Trend Line**: Shows the overall relationship between volatility and error.
            - **Correlation Value**: A value close to 1 means errors increase significantly with volatility, while values near 0 suggest little relationship.
            - **Volatility Clusters**: Look for clusters of points at specific volatility levels - these can reveal market regimes.
            - **Outliers**: Points far above the trend line represent predictions that performed exceptionally poorly relative to the volatility level.
            
            **Actionable insights:**
            - If correlation is high (>0.5), consider using separate models for high/low volatility periods.
            - If most errors occur in specific volatility ranges, those market conditions might require model adjustments.
            - Predictions made during periods with volatility above 0.3 (annualized) typically have higher uncertainty.
            """)
    
    with tab5:
        st.header("Model Evaluation Dashboard")
        st.markdown("""
        This tab provides comprehensive insights into model performance across different stocks and over time.
        
        **Key Features:**
        - **Performance Summary**: Overall metrics showing model accuracy across all stocks
        - **Stock Comparison**: Compare how models perform on different stocks to identify strengths and weaknesses
        - **Time-based Analysis**: Track model improvements and performance trends over time
        - **Raw Data Access**: View the underlying evaluation data for detailed analysis
        
        Use these insights to identify which models are performing best, where improvements are needed, and how model performance evolves over time. This information is valuable for model selection, refinement, and deployment decisions.
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
                    
                    # Add more detailed interpretation guidance
                    st.markdown("""
                    **How to interpret this graph:**
                    
                    - **Time Progression**: The x-axis shows model training dates from oldest (left) to newest (right).
                    - **Multiple Metrics**: Solid lines typically represent accuracy metrics (R²) where higher is better, while dashed lines represent error metrics (RMSE, MAE) where lower is better.
                    - **Different Stocks**: Each color represents a different stock symbol if multiple stocks are being evaluated.
                    - **Convergence**: Lines that flatten over time suggest the model is reaching its performance limit.
                    - **Sudden Changes**: Sharp improvements or degradations can indicate significant model changes or data shifts.
                    
                    **What to look for:**
                    - **Consistent Improvement**: Accuracy metrics should trend upward and error metrics downward over time.
                    - **Seasonality**: Cyclical patterns in performance may indicate seasonal effects in the market.
                    - **Diverging Performance**: If some stocks improve while others worsen, it may indicate the model is becoming specialized.
                    - **Performance Plateaus**: Extended flat periods suggest a need for new features or modeling approaches.
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

# Debug section
if debug_mode and not df.empty:
    st.subheader("⚠️ Debug Data")
    st.markdown("**This section shows raw data for debugging purposes**")
    
    # Display date information
    st.write("### Date Information")
    date_info = {
        "Date Column Type": str(df['date'].dtype),
        "First Date": str(df['date'].min()),
        "Last Date": str(df['date'].max()),
        "Total Days": len(df['date'].unique()),
        "Number of Data Points": len(df)
    }
    st.json(date_info)
    
    # Show raw data
    st.write("### Raw Data (First 10 rows)")
    st.dataframe(df.head(10))
    
    # Show last 10 data points
    st.write("### Raw Data (Last 10 rows)")
    st.dataframe(df.tail(10))
    
    # Plot dates to check for gaps
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        date_df = df.copy()
        date_df['date'] = pd.to_datetime(date_df['date'])
    else:
        date_df = df.copy()
    
    date_df = date_df.sort_values('date')
    date_df['day_number'] = range(len(date_df))
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=date_df['date'],
            y=date_df['day_number'],
            mode='markers+lines',
            name='Date Sequence'
        )
    )
    
    fig.update_layout(
        title="Date Sequence Check (Discontinuities may indicate missing dates)",
        xaxis_title="Date",
        yaxis_title="Sequential Day Number",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display date differences
    date_df['next_date'] = date_df['date'].shift(-1)
    date_df['days_between'] = (date_df['next_date'] - date_df['date']).dt.total_seconds() / (60*60*24)
    date_df = date_df.dropna(subset=['days_between'])
    
    if len(date_df) > 0:
        avg_days = date_df['days_between'].mean()
        max_days = date_df['days_between'].max()
        
        st.write(f"Average days between data points: {avg_days:.2f}")
        st.write(f"Maximum days between data points: {max_days:.2f}")
        
        if max_days > avg_days * 1.5:
            st.warning(f"There may be gaps in the data - maximum gap ({max_days:.2f} days) is significantly higher than average ({avg_days:.2f} days)")
            # Show gaps
            gaps = date_df[date_df['days_between'] > avg_days * 1.5][['date', 'next_date', 'days_between']]
            if not gaps.empty:
                st.write("### Data Gaps")
                st.dataframe(gaps)
    
    # Add detailed analysis of key data points to check for shifts
    st.write("### Data Point Analysis (Check for Shifts)")
    
    # Get original dataframe sorted by date
    df_sorted = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_sorted['date']):
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])
    df_sorted = df_sorted.sort_values('date')
    
    # Create mini dataframes for first and last 3 points
    first_points = df_sorted.head(3).copy()
    last_points = df_sorted.tail(3).copy()
    
    # Display first and last points with extra formatting
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("First 3 Data Points:")
        st.dataframe(first_points[['date', 'actual_price', 'predicted_price']])
        
        # Calculate day gaps at the beginning
        if len(first_points) > 1:
            first_points['next_date'] = first_points['date'].shift(-1)
            first_points['days_gap'] = (first_points['next_date'] - first_points['date']).dt.total_seconds() / (60*60*24)
            first_gaps = first_points.dropna(subset=['days_gap'])[['date', 'next_date', 'days_gap']]
            if not first_gaps.empty:
                st.write("Day gaps between first points:")
                st.dataframe(first_gaps)
    
    with col2:
        st.write("Last 3 Data Points:")
        st.dataframe(last_points[['date', 'actual_price', 'predicted_price']])
        
        # Calculate day gaps at the end
        if len(last_points) > 1:
            last_points_gaps = last_points.copy()
            last_points_gaps['prev_date'] = last_points_gaps['date'].shift(1)
            last_points_gaps['days_gap'] = (last_points_gaps['date'] - last_points_gaps['prev_date']).dt.total_seconds() / (60*60*24)
            last_gaps = last_points_gaps.dropna(subset=['days_gap'])[['prev_date', 'date', 'days_gap']]
            if not last_gaps.empty:
                st.write("Day gaps between last points:")
                st.dataframe(last_gaps)

# Run the app with: streamlit run visualizations/prediction_dashboard.py 