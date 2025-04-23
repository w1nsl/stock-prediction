import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_batch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection parameters
DB_PARAMS = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

def get_db_connection():
    """
    Create a database connection using the parameters from environment variables.
    
    Returns:
        psycopg2 connection object
    """
    return psycopg2.connect(**DB_PARAMS)

def pull_stock_data(stock_symbol: str) -> pd.DataFrame:
    """
    Pull all available stock data for a given stock symbol.
    
    Args:
        stock_symbol: Stock symbol to pull data for
        
    Returns:
        DataFrame containing stock data
    """
    try:
        conn = get_db_connection()
        
        query = f"""
        SELECT * FROM merged_stocks_new 
        WHERE stock_symbol = '{stock_symbol}'
        ORDER BY date
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty:
            raise ValueError(f"No data found for {stock_symbol}")
            
        return df
        
    except Exception as e:
        print(f"Error pulling stock data: {str(e)}")
        raise

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on stock data.
    
    Args:
        df: DataFrame containing raw stock data
        
    Returns:
        DataFrame with engineered features
    """
    try:
        # Make a copy to avoid modifying the original dataframe
        df_features = df.copy()
        
        # Ensure date is in datetime format
        df_features['date'] = pd.to_datetime(df_features['date'])
        
        # Sort by date
        df_features = df_features.sort_values('date')
        
        # Create target variable (1-day future price)
        df_features['target'] = df_features['adj_close'].shift(-1)
        
        # Create binary has_sentiment feature (1 if there are articles, 0 otherwise)
        df_features['has_sentiment'] = (df_features['article_count'] > 0).astype(int)
        
        # Create lag features
        for lag in [1, 2, 3, 5]:
            df_features[f'adj_close_lag_{lag}'] = df_features['adj_close'].shift(lag)
            
        for lag in [1, 2, 3]:
            df_features[f'volume_lag_{lag}'] = df_features['volume'].shift(lag)
            df_features[f'daily_sentiment_lag_{lag}'] = df_features['daily_sentiment'].shift(lag)
        
        # Create rolling mean features
        for window in [3, 7, 14]:
            df_features[f'adj_close_rollmean_{window}'] = df_features['adj_close'].rolling(window=window).mean()
            df_features[f'daily_sentiment_rollmean_{window}'] = df_features['daily_sentiment'].rolling(window=window).mean()
            
        df_features['volume_rollmean_3'] = df_features['volume'].rolling(window=3).mean()
        
        # Create rolling standard deviation features
        df_features['adj_close_rollstd_14'] = df_features['adj_close'].rolling(window=14).std()
        df_features['volume_rollstd_14'] = df_features['volume'].rolling(window=14).std()
        df_features['daily_sentiment_rollstd_14'] = df_features['daily_sentiment'].rolling(window=14).std()
        
        # Create Average True Range (ATR) - a volatility indicator
        df_features['high_low'] = df_features['high_price'] - df_features['low_price']
        df_features['high_close'] = abs(df_features['high_price'] - df_features['adj_close'].shift(1))
        df_features['low_close'] = abs(df_features['low_price'] - df_features['adj_close'].shift(1))
        df_features['true_range'] = df_features[['high_low', 'high_close', 'low_close']].max(axis=1)
        df_features['atr_7d'] = df_features['true_range'].rolling(window=7).mean()
        
        # Create sentiment range features
        df_features['sentiment_max'] = df_features['daily_sentiment'].rolling(window=7).max()
        df_features['sentiment_min'] = df_features['daily_sentiment'].rolling(window=7).min()
        df_features['sentiment_range'] = df_features['sentiment_max'] - df_features['sentiment_min']
        
        # Drop temporary columns
        df_features = df_features.drop(['high_low', 'high_close', 'low_close', 'true_range'], axis=1)
        
        # Drop rows with NaN values (due to lag and rolling calculations)
        df_features = df_features.dropna()
        
        return df_features
        
    except Exception as e:
        print(f"Error engineering features: {str(e)}")
        raise

def load_features_to_db(df: pd.DataFrame, table_name: str = "ml_features"):
    """
    Load engineered features to database.
    
    Args:
        df: DataFrame containing engineered features
        table_name: Name of the table to store features
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create table if it does not exist
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            date DATE,
            stock_symbol TEXT,
            target DOUBLE PRECISION,
            has_sentiment INTEGER,
            daily_sentiment_rollmean_7 DOUBLE PRECISION,
            daily_sentiment_lag_3 DOUBLE PRECISION,
            volume_rollmean_3 DOUBLE PRECISION,
            volume_lag_2 DOUBLE PRECISION,
            adj_close_lag_3 DOUBLE PRECISION,
            adj_close_rollmean_3 DOUBLE PRECISION,
            volume_rollstd_14 DOUBLE PRECISION,
            adj_close_lag_5 DOUBLE PRECISION,
            volume_lag_3 DOUBLE PRECISION,
            adj_close_rollmean_14 DOUBLE PRECISION,
            adj_close_lag_1 DOUBLE PRECISION,
            adj_close_rollstd_14 DOUBLE PRECISION,
            adj_close_lag_2 DOUBLE PRECISION,
            daily_sentiment_lag_1 DOUBLE PRECISION,
            fed_funds_rate DOUBLE PRECISION,
            volume_lag_1 DOUBLE PRECISION,
            daily_sentiment_lag_2 DOUBLE PRECISION,
            daily_sentiment_rollmean_14 DOUBLE PRECISION,
            sentiment_max DOUBLE PRECISION,
            adj_close DOUBLE PRECISION,
            article_count INTEGER,
            sentiment_min DOUBLE PRECISION,
            adj_close_rollmean_7 DOUBLE PRECISION,
            atr_7d DOUBLE PRECISION,
            daily_sentiment_rollmean_3 DOUBLE PRECISION,
            sentiment_range DOUBLE PRECISION,
            daily_sentiment_rollstd_14 DOUBLE PRECISION,
            PRIMARY KEY (date, stock_symbol)
        )
        """
        cursor.execute(create_table_query)
        conn.commit()
        
        # Prepare records for batch insert
        records = [
            (
                row['date'],
                row['stock_symbol'],
                row['target'],
                row['has_sentiment'],
                row['daily_sentiment_rollmean_7'],
                row['daily_sentiment_lag_3'],
                row['volume_rollmean_3'],
                row['volume_lag_2'],
                row['adj_close_lag_3'],
                row['adj_close_rollmean_3'],
                row['volume_rollstd_14'],
                row['adj_close_lag_5'],
                row['volume_lag_3'],
                row['adj_close_rollmean_14'],
                row['adj_close_lag_1'],
                row['adj_close_rollstd_14'],
                row['adj_close_lag_2'],
                row['daily_sentiment_lag_1'],
                row['fed_funds_rate'],
                row['volume_lag_1'],
                row['daily_sentiment_lag_2'],
                row['daily_sentiment_rollmean_14'],
                row['sentiment_max'],
                row['adj_close'],
                row['article_count'],
                row['sentiment_min'],
                row['adj_close_rollmean_7'],
                row['atr_7d'],
                row['daily_sentiment_rollmean_3'],
                row['sentiment_range'],
                row['daily_sentiment_rollstd_14']
            )
            for _, row in df.iterrows()
        ]
        
        # Insert data using batch execution with ON CONFLICT DO UPDATE
        insert_query = f"""
        INSERT INTO {table_name} (
            date, stock_symbol, target, has_sentiment, daily_sentiment_rollmean_7, 
            daily_sentiment_lag_3, volume_rollmean_3, volume_lag_2, adj_close_lag_3, 
            adj_close_rollmean_3, volume_rollstd_14, adj_close_lag_5, volume_lag_3, 
            adj_close_rollmean_14, adj_close_lag_1, adj_close_rollstd_14, adj_close_lag_2, 
            daily_sentiment_lag_1, fed_funds_rate, volume_lag_1, daily_sentiment_lag_2, 
            daily_sentiment_rollmean_14, sentiment_max, adj_close, article_count, 
            sentiment_min, adj_close_rollmean_7, atr_7d, daily_sentiment_rollmean_3, 
            sentiment_range, daily_sentiment_rollstd_14
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON CONFLICT (date, stock_symbol) DO UPDATE SET
            target = EXCLUDED.target,
            has_sentiment = EXCLUDED.has_sentiment,
            daily_sentiment_rollmean_7 = EXCLUDED.daily_sentiment_rollmean_7,
            daily_sentiment_lag_3 = EXCLUDED.daily_sentiment_lag_3,
            volume_rollmean_3 = EXCLUDED.volume_rollmean_3,
            volume_lag_2 = EXCLUDED.volume_lag_2,
            adj_close_lag_3 = EXCLUDED.adj_close_lag_3,
            adj_close_rollmean_3 = EXCLUDED.adj_close_rollmean_3,
            volume_rollstd_14 = EXCLUDED.volume_rollstd_14,
            adj_close_lag_5 = EXCLUDED.adj_close_lag_5,
            volume_lag_3 = EXCLUDED.volume_lag_3,
            adj_close_rollmean_14 = EXCLUDED.adj_close_rollmean_14,
            adj_close_lag_1 = EXCLUDED.adj_close_lag_1,
            adj_close_rollstd_14 = EXCLUDED.adj_close_rollstd_14,
            adj_close_lag_2 = EXCLUDED.adj_close_lag_2,
            daily_sentiment_lag_1 = EXCLUDED.daily_sentiment_lag_1,
            fed_funds_rate = EXCLUDED.fed_funds_rate,
            volume_lag_1 = EXCLUDED.volume_lag_1,
            daily_sentiment_lag_2 = EXCLUDED.daily_sentiment_lag_2,
            daily_sentiment_rollmean_14 = EXCLUDED.daily_sentiment_rollmean_14,
            sentiment_max = EXCLUDED.sentiment_max,
            adj_close = EXCLUDED.adj_close,
            article_count = EXCLUDED.article_count,
            sentiment_min = EXCLUDED.sentiment_min,
            adj_close_rollmean_7 = EXCLUDED.adj_close_rollmean_7,
            atr_7d = EXCLUDED.atr_7d,
            daily_sentiment_rollmean_3 = EXCLUDED.daily_sentiment_rollmean_3,
            sentiment_range = EXCLUDED.sentiment_range,
            daily_sentiment_rollstd_14 = EXCLUDED.daily_sentiment_rollstd_14
        """
        
        execute_batch(cursor, insert_query, records)
        conn.commit()
        print(f"Inserted {len(records)} records into the database.")
        
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error inserting data: {e}")
        raise
    finally:
        if conn:
            conn.close()

def initialize_features_for_stock(stock_symbol: str):
    """
    Initialize the database with engineered features for a specific stock.
    This pulls all available data, engineers features, and loads them into the database.
    
    Args:
        stock_symbol: Stock symbol to initialize features for
    """
    try:
        print(f"Initializing features for {stock_symbol}...")
        
        # Pull all available stock data
        print("Pulling stock data...")
        df = pull_stock_data(stock_symbol)
        
        # Engineer features
        print("Engineering features...")
        df_engineered = engineer_features(df)
        
        # Load features to database
        print("Loading features to database...")
        load_features_to_db(df_engineered)
        
        print(f"Successfully initialized features for {stock_symbol}")
        
    except Exception as e:
        print(f"Error initializing features for {stock_symbol}: {e}")
        raise

def initialize_features_for_all_stocks():
    """
    Initialize features for all stocks in the database.
    """
    conn = None
    try:
        # Get list of unique stock symbols from the database
        conn = get_db_connection()
        cursor = conn.cursor()

        stock_symbols = [
        "ADBE", "CMCSA", "QCOM", 
        "GOOG",
         "PEP",
        "SBUX", "COST", "AMD", 
        "INTC", 
        "PYPL"
    ]

        
        conn.close()
        
        # Initialize features for each stock
        for symbol in stock_symbols:
            initialize_features_for_stock(symbol)
            
    except Exception as e:
        print(f"Error initializing features for all stocks: {e}")
        raise
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    initialize_features_for_all_stocks() 