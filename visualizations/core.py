import pandas as pd
import os
from threading import RLock

# Configure matplotlib properly for non-GUI environments like Streamlit Cloud
import matplotlib
matplotlib.use('Agg')  # Must be called before pyplot
import matplotlib.pyplot as plt

# Create a lock for thread-safe matplotlib operations
matplotlib_lock = RLock()

import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine
from typing import Dict, List, Tuple, Optional, Union
import joblib
import io
import base64

# Import our direct data loader (with fallback to sample data)
try:
    from visualizations.direct_data_loader import load_stock_data_from_dags
    HAS_DIRECT_LOADER = True
except ImportError:
    print("Direct data loader not available, falling back to database or sample data")
    HAS_DIRECT_LOADER = False

# Load environment variables
load_dotenv()

# Database connection
def get_db_connection():
    """Connect to the PostgreSQL database"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        port=int(os.getenv('DB_PORT')),
        sslmode=os.getenv('DB_SSLMODE')
    )

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
        conn.close()
        
        df.attrs['data_source'] = 'database'  # Mark data source
        return df
    except Exception as e:
        print(f"Error loading data from database: {e}")
        print("Generating sample data for demonstration")
        
        # Generate sample data
        if start_date is None:
            start_date = '2023-01-01'
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date)
        
        # Default to AAPL if no stock specified
        if stock_symbol is None:
            stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
        else:
            stocks = [stock_symbol]
            
        # Create sample dataframe
        rows = []
        for stock in stocks:
            # Base price and parameters differ by stock
            if stock == 'AAPL':
                base_price = 150
                volatility = 5
            elif stock == 'MSFT':
                base_price = 300
                volatility = 8
            elif stock == 'GOOG':
                base_price = 130
                volatility = 6
            elif stock == 'AMZN':
                base_price = 120
                volatility = 7
            else:
                base_price = 200
                volatility = 10
                
            # Generate stock price data
            np.random.seed(42 + hash(stock) % 100)  # Different seed per stock
            price_trend = np.linspace(base_price, base_price * 1.3, len(dates))
            close_prices = price_trend + np.random.normal(0, volatility, len(dates))
            
            for i, date in enumerate(dates):
                close_price = close_prices[i]
                open_price = close_price * (1 + np.random.normal(0, 0.01))
                high_price = max(close_price, open_price) * (1 + abs(np.random.normal(0, 0.01)))
                low_price = min(close_price, open_price) * (1 - abs(np.random.normal(0, 0.01)))
                volume = int(np.random.normal(5000000, 2000000))
                
                # Generate sentiment data
                sentiment = np.random.normal(0.1, 0.5)  # Slightly positive bias
                article_count = int(np.random.poisson(5))  # Random article count
                positive_ratio = max(0, min(1, 0.5 + sentiment * 0.3))
                negative_ratio = max(0, min(1, 0.5 - sentiment * 0.3))
                neutral_ratio = max(0, min(1, 1 - positive_ratio - negative_ratio))
                
                # Generate economic data
                gdp = 23000 + np.random.normal(0, 100)
                real_gdp = 20000 + np.random.normal(0, 100)
                unemployment_rate = 3.5 + np.random.normal(0, 0.2)
                cpi = 300 + np.random.normal(0, 5)
                fed_funds_rate = 4.5 + np.random.normal(0, 0.1)
                sp500 = 4200 + np.random.normal(0, 50)
                
                rows.append({
                    'date': date.date(),
                    'stock_symbol': stock,
                    'open_price': open_price,
                    'high_price': high_price,
                    'low_price': low_price,
                    'close_price': close_price,
                    'adj_close': close_price,
                    'volume': volume,
                    'daily_sentiment': sentiment,
                    'article_count': article_count,
                    'sentiment_std': abs(np.random.normal(0, 0.2)),
                    'positive_ratio': positive_ratio,
                    'negative_ratio': negative_ratio,
                    'neutral_ratio': neutral_ratio,
                    'sentiment_median': sentiment + np.random.normal(0, 0.1),
                    'sentiment_min': sentiment - abs(np.random.normal(0, 0.3)),
                    'sentiment_max': sentiment + abs(np.random.normal(0, 0.3)),
                    'sentiment_range': abs(np.random.normal(0, 0.6)),
                    'gdp': gdp,
                    'real_gdp': real_gdp,
                    'unemployment_rate': unemployment_rate,
                    'cpi': cpi,
                    'fed_funds_rate': fed_funds_rate,
                    'sp500': sp500
                })
                
        df = pd.DataFrame(rows)
        if stock_symbol:
            df = df[df['stock_symbol'] == stock_symbol]
        
        # Mark as sample data
        df.attrs['data_source'] = 'sample'
            
        return df

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
    Create a heatmap of correlations between stock price, sentiment, and economic indicators
    
    Args:
        df: DataFrame containing merged data
        stock_symbol: Stock ticker symbol (if None, use all data)
        
    Returns:
        Matplotlib figure object
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
    
    # Use the matplotlib lock for thread safety
    with matplotlib_lock:
        # Create heatmap - using explicit figure creation as recommended
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        
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
    
    # No plt.show() - we return the figure object directly
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

def generate_stock_visualizations(stock_symbol, start_date=None, end_date=None):
    """
    Generate and save various visualizations for a specific stock
    
    Args:
        stock_symbol: Stock ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format (optional)
        end_date: End date in 'YYYY-MM-DD' format (optional)
    """
    # Load data
    df = load_data_from_db(stock_symbol, start_date, end_date)
    
    if not df.empty:
        
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
        # Save figure using the object directly instead of plt.savefig
        corr_fig.savefig(f"visualizations/static/{stock_symbol}_correlation.png")
        
        # 4. Economic dashboard
        econ_fig = plot_economic_dashboard(df, stock_symbol)
        econ_fig.write_html(f"visualizations/static/{stock_symbol}_economic.html")
        
        print("Visualizations created successfully!")

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
        # Save figure using the object directly instead of plt.savefig
        corr_fig.savefig(f"visualizations/static/{stock_symbol}_correlation.png")
        
        # 4. Economic dashboard
        econ_fig = plot_economic_dashboard(df, stock_symbol)
        econ_fig.write_html(f"visualizations/static/{stock_symbol}_economic.html")
        
        print("Visualizations created successfully!") 