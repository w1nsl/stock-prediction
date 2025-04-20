import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import psycopg2
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from visualizations.core import get_db_connection
from sqlalchemy import create_engine, text
from typing import Dict, List, Tuple, Optional, Union

# Load environment variables
load_dotenv()

# Feature descriptions for feature importance visualization
FEATURE_DESCRIPTIONS = {
    'volume': 'Trading volume for the stock',
    'daily_sentiment': 'Average sentiment score from news articles for the day',
    'article_count': 'Number of news articles published about the stock',
    'positive_ratio': 'Ratio of positive sentiment articles to total articles',
    'negative_ratio': 'Ratio of negative sentiment articles to total articles',
    'neutral_ratio': 'Ratio of neutral sentiment articles to total articles',
    'real_gdp': 'Real Gross Domestic Product, a measure of economic activity',
    'unemployment_rate': 'Current unemployment rate percentage',
    'cpi': 'Consumer Price Index, a measure of inflation',
    'fed_funds_rate': 'Federal Reserve funds rate, a key interest rate',
    'return_1d': 'Stock price return over the previous 1 day',
    'return_3d': 'Stock price return over the previous 3 days',
    'return_5d': 'Stock price return over the previous 5 days',
    'ma7': 'Moving average of stock price over 7 days',
    'rsi': 'Relative Strength Index, a momentum oscillator',
    'volatility_7d': 'Standard deviation of returns over 7 days',
    'volume_ma5': 'Moving average of volume over 5 days',
    'volume_change': 'Percentage change in volume from previous day',
    'sentiment_volume': 'Product of sentiment score and trading volume',
    'sentiment_ma3': 'Moving average of sentiment over 3 days',
    'high_news_day': 'Binary indicator for days with abnormally high news volume',
    'fed_rate_increase': 'Binary indicator for days with Fed rate increases',
    'day_sin': 'Sine transformation of day of week (cyclical feature)',
    'day_cos': 'Cosine transformation of day of week (cyclical feature)',
    'month_end': 'Binary indicator for last trading day of month'
}

# Model-specific interpretations
MODEL_INTERPRETATIONS = {
    'random_forest': """
    Random Forest measures feature importance based on how much each feature decreases impurity across all trees in the forest.
    Higher values indicate features that are more useful for making accurate predictions.
    """,
    
    'lightgbm': """
    LightGBM calculates feature importance based on the number of times a feature is used to split the data across all trees.
    Features with higher importance scores are used more frequently in the model's decision process.
    """
}

def load_predictions(stock_symbol, start_date=None, end_date=None):
    """
    Load prediction data from database
    
    Note: This assumes you have a predictions table in your database.
    You may need to modify this function based on your actual database schema.
    """
    try:
        conn = get_db_connection()
        
        query = """
        SELECT p.date, p.stock_symbol, p.predicted_price, m.close_price as actual_price
        FROM stock_predictions p
        JOIN merged_stocks_new m 
            ON p.date = m.date AND p.stock_symbol = m.stock_symbol
        WHERE p.stock_symbol = %s
        """
        params = [stock_symbol]
        
        if start_date:
            query += " AND p.date >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND p.date <= %s"
            params.append(end_date)
        
        query += " ORDER BY p.date"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            raise Exception("No prediction data found for this stock and date range")
            
        df.attrs['data_source'] = 'database'
        return df
        
    except Exception as e:
        print(f"Error loading prediction data: {e}")
        print("Generating sample prediction data for demonstration")
        
        # Generate sample data for demonstration
        if start_date is None:
            start_date = '2023-01-01'
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date)
        
        # Create dummy prediction data
        np.random.seed(42)  # For reproducibility
        
        # Set base price based on stock
        if stock_symbol == 'AAPL':
            base_price = 150
            volatility = 5
        elif stock_symbol == 'MSFT':
            base_price = 300
            volatility = 8
        elif stock_symbol == 'GOOG':
            base_price = 130
            volatility = 6
        elif stock_symbol == 'AMZN':
            base_price = 120
            volatility = 7
        else:
            base_price = 200
            volatility = 10
            
        # Generate sample trend with noise
        price_trend = np.linspace(base_price, base_price * 1.3, len(dates))
        actual_prices = price_trend + np.random.normal(0, volatility, len(dates))
        
        # Predictions are actual prices with added error (more error for further dates)
        error_scale = np.linspace(volatility * 0.2, volatility * 0.6, len(dates))
        predicted_prices = actual_prices + np.random.normal(0, error_scale, len(dates))
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'stock_symbol': stock_symbol,
            'actual_price': actual_prices,
            'predicted_price': predicted_prices
        })
        
        # Mark as sample data
        df.attrs['data_source'] = 'sample'
        
        return df

def plot_prediction_comparison(df, stock_symbol, window_size=10):
    """
    Create a comparative visualization of actual vs. predicted prices
    
    Args:
        df: DataFrame with date, predicted_price, and actual_price columns
        stock_symbol: Stock ticker symbol
        window_size: Window size for rolling average
    """
    fig = go.Figure()
    
    # Add actual price line
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['actual_price'],
            mode='lines',
            name='Actual Price',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add predicted price line
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['predicted_price'],
            mode='lines',
            name='Predicted Price',
            line=dict(color='red', width=2)
        )
    )
    
    # Calculate and add rolling average of actual price
    if len(df) >= window_size:
        df['rolling_avg'] = df['actual_price'].rolling(window=window_size).mean()
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['rolling_avg'],
                mode='lines',
                name=f'{window_size}-Day Rolling Avg',
                line=dict(color='green', width=1.5, dash='dash')
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"{stock_symbol} - Actual vs. Predicted Price",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600
    )
    
    return fig

def plot_error_analysis(df, stock_symbol):
    """
    Create a visualization of prediction errors over time
    
    Args:
        df: DataFrame with date, predicted_price, and actual_price columns
        stock_symbol: Stock ticker symbol
    """
    # Calculate errors
    df = df.copy()
    df['error'] = df['actual_price'] - df['predicted_price']
    df['abs_error'] = abs(df['error'])
    df['pct_error'] = (df['error'] / df['actual_price']) * 100
    
    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "Prediction Error Over Time",
            "Absolute Error Over Time",
            "Percentage Error Over Time"
        ),
        row_heights=[0.33, 0.33, 0.33]
    )
    
    # Add error trace
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['error'],
            mode='lines',
            name='Error',
            line=dict(color='purple')
        ),
        row=1, col=1
    )
    
    # Add zero line
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=[0] * len(df),
            mode='lines',
            name='Zero Error',
            line=dict(color='black', dash='dash', width=1)
        ),
        row=1, col=1
    )
    
    # Add absolute error trace
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['abs_error'],
            mode='lines',
            name='Absolute Error',
            line=dict(color='orange'),
            fill='tozeroy'
        ),
        row=2, col=1
    )
    
    # Add percentage error trace
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['pct_error'],
            mode='lines',
            name='Percentage Error',
            line=dict(color='green'),
            fill='tozeroy'
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{stock_symbol} - Prediction Error Analysis",
        height=900,
        showlegend=False
    )
    
    return fig

def plot_error_distribution(df, stock_symbol):
    """
    Create a visualization of the error distribution
    
    Args:
        df: DataFrame with date, predicted_price, and actual_price columns
        stock_symbol: Stock ticker symbol
    """
    # Calculate errors
    df = df.copy()
    df['error'] = df['actual_price'] - df['predicted_price']
    df['pct_error'] = (df['error'] / df['actual_price']) * 100
    
    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Error Distribution",
            "Percentage Error Distribution"
        ),
        column_widths=[0.5, 0.5]
    )
    
    # Add error histogram
    fig.add_trace(
        go.Histogram(
            x=df['error'],
            name='Error',
            marker_color='purple',
            opacity=0.7,
            nbinsx=30
        ),
        row=1, col=1
    )
    
    # Add percentage error histogram
    fig.add_trace(
        go.Histogram(
            x=df['pct_error'],
            name='Percentage Error',
            marker_color='green',
            opacity=0.7,
            nbinsx=30
        ),
        row=1, col=2
    )
    
    # Add vertical lines at zero
    fig.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title=f"{stock_symbol} - Error Distribution Analysis",
        height=500,
        showlegend=False
    )
    
    return fig

def calculate_metrics(df):
    """Calculate prediction performance metrics"""
    # Basic error metrics
    rmse = np.sqrt(mean_squared_error(df['actual_price'], df['predicted_price']))
    mae = mean_absolute_error(df['actual_price'], df['predicted_price'])
    r2 = r2_score(df['actual_price'], df['predicted_price'])
    
    # Mean absolute percentage error
    mape = np.mean(np.abs((df['actual_price'] - df['predicted_price']) / df['actual_price'])) * 100
    
    # Directional accuracy (correctly predicting up/down movements)
    df['actual_direction'] = df['actual_price'].diff().apply(lambda x: 1 if x >= 0 else 0)
    df['predicted_direction'] = df['predicted_price'].diff().apply(lambda x: 1 if x >= 0 else 0)
    direction_match = (df['actual_direction'] == df['predicted_direction']).sum()
    direction_accuracy = direction_match / (len(df) - 1) * 100  # -1 because diff loses one row
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape,
        'Direction Accuracy': direction_accuracy
    }

def plot_accuracy_vs_horizon(df, stock_symbol, max_days=10):
    """
    Analyze how prediction accuracy changes with prediction horizon
    
    Args:
        df: DataFrame with date, predicted_price, and actual_price columns
        stock_symbol: Stock ticker symbol
        max_days: Maximum prediction horizon to analyze
    """
    # Initialize containers for metrics
    horizons = list(range(1, max_days+1))
    rmse_values = []
    mae_values = []
    mape_values = []
    dir_acc_values = []
    
    # Calculate metrics for each horizon
    for horizon in horizons:
        # Create shifted predictions (as if they were made n days in advance)
        df_shifted = df.copy()
        df_shifted['predicted_price'] = df_shifted['predicted_price'].shift(-horizon)
        df_shifted = df_shifted.dropna()
        
        if len(df_shifted) > 0:
            # Calculate metrics
            metrics = calculate_metrics(df_shifted)
            rmse_values.append(metrics['RMSE'])
            mae_values.append(metrics['MAE'])
            mape_values.append(metrics['MAPE'])
            dir_acc_values.append(metrics['Direction Accuracy'])
        else:
            # Not enough data for this horizon
            rmse_values.append(np.nan)
            mae_values.append(np.nan)
            mape_values.append(np.nan)
            dir_acc_values.append(np.nan)
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "RMSE vs Prediction Horizon",
            "MAE vs Prediction Horizon",
            "MAPE vs Prediction Horizon",
            "Direction Accuracy vs Prediction Horizon"
        )
    )
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=horizons,
            y=rmse_values,
            mode='lines+markers',
            name='RMSE',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=horizons,
            y=mae_values,
            mode='lines+markers',
            name='MAE',
            line=dict(color='blue')
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=horizons,
            y=mape_values,
            mode='lines+markers',
            name='MAPE',
            line=dict(color='green')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=horizons,
            y=dir_acc_values,
            mode='lines+markers',
            name='Direction Accuracy',
            line=dict(color='purple')
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"{stock_symbol} - Prediction Accuracy vs Horizon",
        height=800,
        showlegend=False
    )
    
    return fig

def plot_performance_by_volatility(df, stock_symbol, window=20):
    """
    Analyze how prediction accuracy relates to market volatility
    
    Args:
        df: DataFrame with date, predicted_price, and actual_price columns
        stock_symbol: Stock ticker symbol
        window: Rolling window size for volatility calculation
    """
    # Calculate volatility
    df = df.copy()
    df['volatility'] = df['actual_price'].pct_change().rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    # Calculate absolute percentage error
    df['pct_error'] = abs((df['actual_price'] - df['predicted_price']) / df['actual_price']) * 100
    
    # Remove NaN values
    df = df.dropna()
    
    # Create scatter plot
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df['volatility'],
            y=df['pct_error'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['date'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Date")
            ),
            text=df['date'].astype(str),
            hovertemplate='Date: %{text}<br>Volatility: %{x:.4f}<br>Prediction Error: %{y:.2f}%'
        )
    )
    
    # Calculate correlation
    correlation = df['volatility'].corr(df['pct_error'])
    
    # Add trend line
    z = np.polyfit(df['volatility'], df['pct_error'], 1)
    p = np.poly1d(z)
    fig.add_trace(
        go.Scatter(
            x=df['volatility'],
            y=p(df['volatility']),
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"{stock_symbol} - Prediction Error vs Volatility (Correlation: {correlation:.4f})",
        xaxis_title="Volatility (Annualized)",
        yaxis_title="Percentage Error (%)",
        height=600
    )
    
    return fig

def sample_predictions_to_csv(df, output_path):
    """Save predictions to CSV for further analysis"""
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

def load_feature_importance():
    """
    Load feature importance data from ML models
    
    Returns:
        dict: Dictionary containing feature importance data for different models
    """
    try:
        # Define feature names based on the models in ml_notebook2.ipynb
        feature_cols = [
            'volume', 'daily_sentiment', 'article_count', 'positive_ratio', 
            'negative_ratio', 'neutral_ratio', 'real_gdp', 'unemployment_rate', 
            'cpi', 'fed_funds_rate', 'return_1d', 'return_3d', 'return_5d', 
            'ma7', 'rsi', 'volatility_7d', 'volume_ma5', 'volume_change', 
            'sentiment_volume', 'sentiment_ma3', 'high_news_day', 
            'fed_rate_increase', 'day_sin', 'day_cos', 'month_end'
        ]
        
        # Example feature importance values for Random Forest
        # These would ideally come from your saved model or database
        rf_importance = [
            0.142, 0.093, 0.078, 0.072, 0.068, 0.062, 0.058, 0.056, 
            0.052, 0.048, 0.042, 0.039, 0.038, 0.032, 0.029, 0.025, 
            0.023, 0.021, 0.019, 0.018, 0.015, 0.014, 0.012, 0.010, 0.004
        ]
        
        # Example feature importance values for LightGBM
        lgb_importance = [
            0.128, 0.112, 0.091, 0.081, 0.072, 0.064, 0.052, 0.049, 
            0.047, 0.043, 0.039, 0.037, 0.035, 0.032, 0.029, 0.026, 
            0.021, 0.018, 0.017, 0.015, 0.014, 0.012, 0.011, 0.009, 0.006
        ]
        
        # Create sorted dataframes for each model
        rf_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_importance
        }).sort_values('importance', ascending=False)
        
        lgb_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': lgb_importance
        }).sort_values('importance', ascending=False)
        
        return {
            'random_forest': rf_df,
            'lightgbm': lgb_df
        }
    except Exception as e:
        print(f"Error loading feature importance data: {e}")
        # Return empty dataframes if there's an error
        return {
            'random_forest': pd.DataFrame(columns=['feature', 'importance']),
            'lightgbm': pd.DataFrame(columns=['feature', 'importance'])
        }

if __name__ == "__main__":
    # Example usage
    stock_symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # For testing without database, create dummy prediction data
    # In real use, you would use load_predictions(stock_symbol, start_date, end_date)
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date)
    
    # Create dummy data
    np.random.seed(42)  # For reproducibility
    actual_prices = np.linspace(150, 200, len(dates)) + np.random.normal(0, 5, len(dates))
    predicted_prices = actual_prices + np.random.normal(0, 10, len(dates))
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'stock_symbol': stock_symbol,
        'actual_price': actual_prices,
        'predicted_price': predicted_prices
    })
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    print("\nPrediction Performance Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    # Create output directory if it doesn't exist
    os.makedirs("visualizations/static", exist_ok=True)
    
    # Create visualizations
    comparison_fig = plot_prediction_comparison(df, stock_symbol)
    comparison_fig.write_html(f"visualizations/static/{stock_symbol}_prediction_comparison.html")
    
    error_fig = plot_error_analysis(df, stock_symbol)
    error_fig.write_html(f"visualizations/static/{stock_symbol}_error_analysis.html")
    
    distribution_fig = plot_error_distribution(df, stock_symbol)
    distribution_fig.write_html(f"visualizations/static/{stock_symbol}_error_distribution.html")
    
    horizon_fig = plot_accuracy_vs_horizon(df, stock_symbol)
    horizon_fig.write_html(f"visualizations/static/{stock_symbol}_accuracy_horizon.html")
    
    volatility_fig = plot_performance_by_volatility(df, stock_symbol)
    volatility_fig.write_html(f"visualizations/static/{stock_symbol}_volatility_analysis.html")
    
    # Save sample predictions to CSV
    os.makedirs("data", exist_ok=True)
    sample_predictions_to_csv(df.head(100), "data/sample_predictions.csv")
    
    print("All visualizations created successfully!") 