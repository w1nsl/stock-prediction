import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import os
from dotenv import load_dotenv
import psycopg2
from datetime import datetime, timedelta
import joblib
from typing import List, Dict, Tuple, Optional, Union, Any
import io
import base64
import time
from psycopg2 import OperationalError

# Import our direct data loader (with fallback to sample data)
try:
    from visualizations.direct_data_loader import load_stock_data_from_dags
    HAS_DIRECT_LOADER = True
except ImportError:
    print("Direct data loader not available, falling back to database or sample data")
    HAS_DIRECT_LOADER = False

# Load environment variables
load_dotenv()

# Database connection pool
# Create a global connection pool that can be reused
_connection_pool = None

def get_db_connection():
    """Connect to the PostgreSQL database with retry logic and connection pooling"""
    global _connection_pool
    max_retries = 3
    retry_delay = 2  # seconds
    
    # If we already have a connection pool, try to get a connection from it
    if _connection_pool is not None:
        try:
            conn = _connection_pool.getconn()
            # Test if connection is still valid
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            print("Reusing connection from pool")
            return conn
        except Exception as e:
            print(f"Error reusing connection: {e}, will create new one")
            # Connection failed, create a new one below
            try:
                _connection_pool.putconn(conn)  # Return failed connection to pool
            except:
                pass  # Ignore errors returning connection
    
    for attempt in range(max_retries):
        try:
            # Try to use Streamlit secrets first (for cloud deployment)
            import streamlit as st
            if hasattr(st, 'secrets') and 'db' in st.secrets:
                print(f"Attempt {attempt+1}/{max_retries}: Using Streamlit secrets for database connection")
                from psycopg2 import pool
                
                # Create a connection pool if we don't have one
                if _connection_pool is None:
                    _connection_pool = pool.ThreadedConnectionPool(
                        minconn=1, 
                        maxconn=10,
                        host=st.secrets.db.host,
                        database=st.secrets.db.name,
                        user=st.secrets.db.user,
                        password=st.secrets.db.password,
                        port=int(st.secrets.db.port),
                        sslmode=st.secrets.db.sslmode,
                        connect_timeout=5  # Reduced timeout to prevent long hangs
                    )
                
                # Get a connection from the pool
                conn = _connection_pool.getconn()
                
                # Test connection with a simple query
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                print("Database connection successful!")
                return conn
            
            # Fallback to environment variables (for local development)
            else:
                print(f"Attempt {attempt+1}/{max_retries}: Using environment variables for database connection")
                from psycopg2 import pool
                
                # Create a connection pool if we don't have one
                if _connection_pool is None:
                    _connection_pool = pool.ThreadedConnectionPool(
                        minconn=1, 
                        maxconn=10,
                        host=os.getenv('DB_HOST'),
                        database=os.getenv('DB_NAME'),
                        user=os.getenv('DB_USER'),
                        password=os.getenv('DB_PASSWORD'),
                        port=int(os.getenv('DB_PORT')),
                        sslmode=os.getenv('DB_SSLMODE'),
                        connect_timeout=5  # Reduced timeout to prevent long hangs
                    )
                
                # Get a connection from the pool
                conn = _connection_pool.getconn()
                
                # Test connection with a simple query
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                print("Database connection successful!")
                return conn
                
        except OperationalError as e:
            print(f"Connection attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("All connection attempts failed")
                raise Exception(f"Database connection failed after {max_retries} attempts: {e}")
        except Exception as e:
            print(f"Unexpected error during database connection: {e}")
            raise Exception(f"Database connection error: {e}")

def close_db_connection(conn):
    """Properly return a connection to the pool"""
    global _connection_pool
    if _connection_pool is not None and conn is not None:
        try:
            _connection_pool.putconn(conn)
            return True
        except Exception as e:
            print(f"Error returning connection to pool: {e}")
    return False

def load_data_from_db(stock_symbol=None, start_date=None, end_date=None):
    """
    Load data from the database
    
    Args:
        stock_symbol: Stock ticker symbol (if None, load all stocks)
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame containing merged data
    """
    # Try to use the direct data loader from DAGs first
    if HAS_DIRECT_LOADER:
        try:
            print("Attempting to load data directly from DAG files...")
            df = load_stock_data_from_dags(stock_symbol, start_date, end_date)
            if not df.empty:
                print(f"Successfully loaded {len(df)} records directly from DAGs")
                return df
            else:
                print("Direct data loader returned empty DataFrame, falling back to database")
        except Exception as e:
            print(f"Error using direct data loader: {e}, falling back to database")

    # If direct data loader failed or is not available, try the database
    try:
        print("Attempting to load data from database...")
        conn = get_db_connection()
        
        if conn is None:
            raise Exception("Failed to establish database connection")
        
        query = "SELECT * FROM merged_stocks_new WHERE 1=1"
        params = []
        
        if stock_symbol:
            query += " AND stock_symbol = %s"
            params.append(stock_symbol)
        
        if start_date:
            query += " AND date >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= %s"
            params.append(end_date)
        
        query += " ORDER BY stock_symbol, date"
        
        df = pd.read_sql_query(query, conn, params=params)
        # Return connection to the pool instead of closing it
        close_db_connection(conn)
        
        if df.empty:
            print(f"Query returned no data for {stock_symbol} from {start_date} to {end_date}")
        else:
            print(f"Successfully loaded {len(df)} records from database")
            
        df.attrs['data_source'] = 'database'  # Mark data source
        return df
    except Exception as e:
        print(f"Error loading data from database: {e}")
        # Raise the exception instead of generating sample data
        raise Exception(f"Failed to load data: {e}")

# 1. Stock Price Candlestick Chart with Volume
def plot_stock_candlestick(df, stock_symbol, title=None):
    """
    Create an interactive candlestick chart with volume subplot
    
    Args:
        df: DataFrame containing stock data
        stock_symbol: Stock ticker symbol
        title: Chart title (optional)
    """
    if title is None:
        title = f"{stock_symbol} Stock Price"
    
    # Filter for the specific stock
    stock_df = df[df['stock_symbol'] == stock_symbol].copy()
    
    # Create subplot figure with 2 rows (price and volume)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=(title, 'Volume'), 
                        row_heights=[0.7, 0.3])
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=stock_df['date'],
            open=stock_df['open_price'],
            high=stock_df['high_price'],
            low=stock_df['low_price'],
            close=stock_df['close_price'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add volume bar chart
    fig.add_trace(
        go.Bar(
            x=stock_df['date'],
            y=stock_df['volume'],
            name="Volume",
            marker_color='rgba(0, 0, 255, 0.5)'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=False,
        xaxis_rangeslider_visible=False
    )
    
    return fig

# 2. Sentiment Analysis Visualization
def plot_sentiment_analysis(df, stock_symbol):
    """
    Create a visualization of sentiment analysis metrics over time
    
    Args:
        df: DataFrame containing sentiment data
        stock_symbol: Stock ticker symbol
    """
    # Filter for the specific stock
    stock_df = df[df['stock_symbol'] == stock_symbol].copy()
    
    # Create subplot with 3 rows
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f"{stock_symbol} Sentiment Score vs. Stock Price",
            "Sentiment Distribution",
            "Article Count"
        ),
        row_heights=[0.5, 0.3, 0.2]
    )
    
    # Row 1: Stock price with sentiment overlay
    fig.add_trace(
        go.Scatter(
            x=stock_df['date'],
            y=stock_df['close_price'],
            name="Close Price",
            line=dict(color='black', width=1)
        ),
        row=1, col=1
    )
    
    # Add secondary y-axis for sentiment
    fig.add_trace(
        go.Scatter(
            x=stock_df['date'],
            y=stock_df['daily_sentiment'],
            name="Sentiment",
            line=dict(color='blue', width=2),
            yaxis="y2"
        ),
        row=1, col=1
    )
    
    # Row 2: Sentiment ratio stacked area chart
    fig.add_trace(
        go.Scatter(
            x=stock_df['date'],
            y=stock_df['positive_ratio'],
            name="Positive",
            stackgroup='sentiment',
            fillcolor='rgba(0, 255, 0, 0.5)'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=stock_df['date'],
            y=stock_df['neutral_ratio'],
            name="Neutral",
            stackgroup='sentiment',
            fillcolor='rgba(200, 200, 200, 0.5)'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=stock_df['date'],
            y=stock_df['negative_ratio'],
            name="Negative",
            stackgroup='sentiment',
            fillcolor='rgba(255, 0, 0, 0.5)'
        ),
        row=2, col=1
    )
    
    # Row 3: Article count
    fig.add_trace(
        go.Bar(
            x=stock_df['date'],
            y=stock_df['article_count'],
            name="Article Count",
            marker_color='rgba(100, 149, 237, 0.8)'
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        yaxis2=dict(
            title="Sentiment Score",
            overlaying="y",
            side="right"
        )
    )
    
    return fig

# 3. Economic Indicators Correlation Heatmap
def plot_correlation_heatmap(df, stock_symbol=None):
    """
    Create a correlation heatmap of stock prices, sentiment, and economic indicators
    
    Args:
        df: DataFrame containing merged data
        stock_symbol: Stock ticker symbol (if None, use all data)
    """
    if stock_symbol:
        df = df[df['stock_symbol'] == stock_symbol].copy()
    
    # Select columns for correlation
    cols_to_correlate = [
        'close_price', 'daily_sentiment', 'article_count',
        'positive_ratio', 'negative_ratio', 'gdp', 'real_gdp', 
        'unemployment_rate', 'cpi', 'fed_funds_rate', 'sp500'
    ]
    
    # Calculate correlation matrix
    corr = df[cols_to_correlate].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    title = "Correlation Between Stock Price, Sentiment, and Economic Indicators"
    if stock_symbol:
        title += f" for {stock_symbol}"
    
    sns.heatmap(
        corr, 
        mask=mask,
        annot=True, 
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    
    return fig

# 4. Combined Economic Indicators Dashboard
def plot_economic_dashboard(df, stock_symbol):
    """
    Create an economic indicators dashboard with stock price overlay
    
    Args:
        df: DataFrame containing merged data
        stock_symbol: Stock ticker symbol
    """
    stock_df = df[df['stock_symbol'] == stock_symbol].copy()
    
    # Create a figure with subplots
    fig = make_subplots(
        rows=4, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Stock Price", "S&P 500",
            "Fed Funds Rate", "Unemployment Rate",
            "CPI", "GDP",
            "Real GDP", "Economic Indicator Comparison"
        )
    )
    
    # Add traces to subplots
    # Stock Price
    fig.add_trace(
        go.Scatter(
            x=stock_df['date'],
            y=stock_df['close_price'],
            name=f"{stock_symbol} Price",
            line=dict(color='black')
        ),
        row=1, col=1
    )
    
    # S&P 500
    fig.add_trace(
        go.Scatter(
            x=stock_df['date'],
            y=stock_df['sp500'],
            name="S&P 500",
            line=dict(color='green')
        ),
        row=1, col=2
    )
    
    # Fed Funds Rate
    fig.add_trace(
        go.Scatter(
            x=stock_df['date'],
            y=stock_df['fed_funds_rate'],
            name="Fed Funds Rate",
            line=dict(color='red')
        ),
        row=2, col=1
    )
    
    # Unemployment Rate
    fig.add_trace(
        go.Scatter(
            x=stock_df['date'],
            y=stock_df['unemployment_rate'],
            name="Unemployment",
            line=dict(color='blue')
        ),
        row=2, col=2
    )
    
    # CPI
    fig.add_trace(
        go.Scatter(
            x=stock_df['date'],
            y=stock_df['cpi'],
            name="CPI",
            line=dict(color='purple')
        ),
        row=3, col=1
    )
    
    # GDP
    fig.add_trace(
        go.Scatter(
            x=stock_df['date'],
            y=stock_df['gdp'],
            name="GDP",
            line=dict(color='orange')
        ),
        row=3, col=2
    )
    
    # Real GDP
    fig.add_trace(
        go.Scatter(
            x=stock_df['date'],
            y=stock_df['real_gdp'],
            name="Real GDP",
            line=dict(color='brown')
        ),
        row=4, col=1
    )
    
    # Normalized comparison of all indicators
    indicators = ['fed_funds_rate', 'unemployment_rate', 'cpi', 'gdp', 'real_gdp']
    for indicator in indicators:
        # Normalize to 0-1 scale for comparison
        normalized = (stock_df[indicator] - stock_df[indicator].min()) / (stock_df[indicator].max() - stock_df[indicator].min())
        fig.add_trace(
            go.Scatter(
                x=stock_df['date'],
                y=normalized,
                name=indicator,
                mode='lines'
            ),
            row=4, col=2
        )
    
    fig.update_layout(
        height=1000,
        title_text=f"Economic Indicators Dashboard for {stock_symbol}",
        showlegend=False,
    )
    
    return fig

# 5. Prediction Performance Visualization
def plot_prediction_performance(actual, predicted, stock_symbol, metric_name='RMSE'):
    """
    Create a visualization of model prediction performance
    
    Args:
        actual: Series of actual values
        predicted: Series of predicted values
        stock_symbol: Stock ticker symbol
        metric_name: Name of the evaluation metric
    """
    # Calculate error metrics
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    
    # Create figure with price and prediction
    fig = go.Figure()
    
    # Add actual price line
    fig.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        )
    )
    
    # Add predicted price line
    fig.add_trace(
        go.Scatter(
            x=predicted.index,
            y=predicted,
            mode='lines',
            name='Predicted',
            line=dict(color='red')
        )
    )
    
    # Add a line for the error
    error = actual - predicted
    fig.add_trace(
        go.Scatter(
            x=actual.index,
            y=error,
            mode='lines',
            name='Error',
            line=dict(color='green', dash='dash'),
            opacity=0.7
        )
    )
    
    # Add annotation with metrics
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text=f"RMSE: {rmse:.4f}<br>MAE: {mae:.4f}",
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.update_layout(
        title=f"Prediction Performance for {stock_symbol}",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# 6. Feature Importance Visualization
def plot_feature_importance(feature_importance: np.ndarray, feature_names: List[str]) -> go.Figure:
    """
    Create a plotly bar chart for feature importance.
    
    Parameters:
    -----------
    feature_importance : np.ndarray
        Array of feature importance values
    feature_names : List[str]
        List of feature names
        
    Returns:
    --------
    go.Figure
        Plotly figure object with feature importance visualization
    """
    # Sort features by importance
    sorted_idx = feature_importance.argsort()
    sorted_importance = feature_importance[sorted_idx]
    sorted_features = np.array(feature_names)[sorted_idx]
    
    # Create horizontal bar chart
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=sorted_features,
            x=sorted_importance,
            orientation='h',
            marker=dict(
                color=sorted_importance,
                colorscale='Viridis',
                colorbar=dict(title="Importance"),
            )
        )
    )
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(autorange="reversed")  # Put highest importance at the top
    )
    
    return fig

def load_feature_importance() -> Dict[str, pd.DataFrame]:
    """
    Load feature importance data from saved model files.
    
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with keys 'random_forest' and 'lightgbm', 
        each containing a DataFrame with columns 'feature' and 'importance'
    """
    load_dotenv()
    
    # Path to model files
    base_path = os.getenv('MODEL_PATH', './models')
    rf_model_path = os.path.join(base_path, 'random_forest_model.joblib')
    lgbm_model_path = os.path.join(base_path, 'lightgbm_model.joblib')
    
    # Initialize empty DataFrames
    rf_importance = pd.DataFrame(columns=['feature', 'importance'])
    lgbm_importance = pd.DataFrame(columns=['feature', 'importance'])
    
    # Feature names from notebooks
    feature_names = [
        'volume', 'daily_sentiment', 'article_count', 'positive_ratio', 
        'negative_ratio', 'neutral_ratio', 'real_gdp', 'unemployment_rate', 
        'cpi', 'fed_funds_rate', 'return_1d', 'return_3d', 'return_5d', 
        'ma7', 'rsi', 'volatility_7d', 'volume_ma5', 'volume_change', 
        'sentiment_volume', 'sentiment_ma3', 'high_news_day', 'fed_rate_increase', 
        'day_sin', 'day_cos', 'month_end'
    ]
    
    try:
        # Try to load Random Forest model and extract feature importance
        if os.path.exists(rf_model_path):
            rf_model = joblib.load(rf_model_path)
            importances = rf_model.feature_importances_
            
            # Create DataFrame with feature names and importance values
            rf_importance = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances
            })
            
            # Sort by importance in descending order
            rf_importance = rf_importance.sort_values('importance', ascending=False).reset_index(drop=True)
    except Exception as e:
        print(f"Error loading Random Forest feature importance: {e}")
    
    try:
        # Try to load LightGBM model and extract feature importance
        if os.path.exists(lgbm_model_path):
            lgbm_model = joblib.load(lgbm_model_path)
            importances = lgbm_model.feature_importances_
            
            # Create DataFrame with feature names and importance values
            lgbm_importance = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances
            })
            
            # Sort by importance in descending order
            lgbm_importance = lgbm_importance.sort_values('importance', ascending=False).reset_index(drop=True)
    except Exception as e:
        print(f"Error loading LightGBM feature importance: {e}")
    
    # For testing/development, generate random data if models don't exist
    if rf_importance.empty and not os.path.exists(rf_model_path):
        print("Random Forest model not found. Generating sample data.")
        np.random.seed(42)
        importances = np.random.rand(len(feature_names))
        importances = importances / np.sum(importances)  # Normalize
        
        rf_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        rf_importance = rf_importance.sort_values('importance', ascending=False).reset_index(drop=True)
    
    if lgbm_importance.empty and not os.path.exists(lgbm_model_path):
        print("LightGBM model not found. Generating sample data.")
        np.random.seed(43)
        importances = np.random.rand(len(feature_names))
        importances = importances / np.sum(importances)  # Normalize
        
        lgbm_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        lgbm_importance = lgbm_importance.sort_values('importance', ascending=False).reset_index(drop=True)
    
    return {
        'random_forest': rf_importance,
        'lightgbm': lgbm_importance
    }

if __name__ == "__main__":
    # Example usage
    stock_symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Load data
    print(f"Loading data for {stock_symbol} from {start_date} to {end_date}...")
    df = load_data_from_db(stock_symbol=stock_symbol, start_date=start_date, end_date=end_date)
    
    if df.empty:
        print("No data found. Please check your database connection or query parameters.")
    else:
        print(f"Data loaded successfully with {len(df)} rows.")
        
        # Create output directory if it doesn't exist
        os.makedirs("visualizations/static", exist_ok=True)
        
        # Create and save visualizations
        # 1. Candlestick chart
        candlestick_fig = plot_stock_candlestick(df, stock_symbol)
        candlestick_fig.write_html(f"visualizations/static/{stock_symbol}_candlestick.html")
        
        # 2. Sentiment analysis visualization
        sentiment_fig = plot_sentiment_analysis(df, stock_symbol)
        sentiment_fig.write_html(f"visualizations/static/{stock_symbol}_sentiment.html")
        
        # 3. Correlation heatmap
        corr_fig = plot_correlation_heatmap(df, stock_symbol)
        plt.savefig(f"visualizations/static/{stock_symbol}_correlation.png")
        
        # 4. Economic dashboard
        econ_fig = plot_economic_dashboard(df, stock_symbol)
        econ_fig.write_html(f"visualizations/static/{stock_symbol}_economic.html")
        
        print("Visualizations created successfully!") 