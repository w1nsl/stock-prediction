import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from dotenv import load_dotenv
from airflow.providers.postgres.hooks.postgres import PostgresHook
from psycopg2.extras import execute_batch

# Load environment variables
load_dotenv()

def pull_stock_data(stock_symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Pull stock data from the merged_stocks_new table for feature engineering based on specified date range.
    Automatically extends the start date to ensure enough historical data for feature engineering.
    
    Args:
        stock_symbol: Stock symbol to pull data for
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame containing stock data
    """
    try:
        pg_hook = PostgresHook(postgres_conn_id='neon_db')
        conn = pg_hook.get_conn()
        
        # First, check if the stock exists and get date range
        check_query = f"""
        SELECT 
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(*) as total_records
        FROM merged_stocks_new 
        WHERE stock_symbol = '{stock_symbol}'
        """
        cursor = conn.cursor()
        cursor.execute(check_query)
        min_date, max_date, total_records = cursor.fetchone()
        print(f"Stock {stock_symbol} data range: {min_date} to {max_date}, total records: {total_records}")
        
        # Calculate extended start date (14 days before requested start date)
        # This ensures we have enough historical data for feature engineering
        extended_start_date = (pd.Timestamp(start_date) - pd.Timedelta(days=20)).strftime('%Y-%m-%d')
        print(f"Extended start date from {start_date} to {extended_start_date} to ensure enough historical data")
        
        # Check records in the specified date range
        range_check_query = f"""
        SELECT COUNT(*) 
        FROM merged_stocks_new 
        WHERE stock_symbol = '{stock_symbol}'
        AND date >= '{extended_start_date}'
        AND date <= '{end_date}'
        """
        cursor.execute(range_check_query)
        range_count = cursor.fetchone()[0]
        print(f"Records in extended date range {extended_start_date} to {end_date}: {range_count}")
        
        # Build main query with extended start date
        query = f"""
        SELECT * FROM merged_stocks_new 
        WHERE stock_symbol = '{stock_symbol}'
        AND date >= '{extended_start_date}'
        AND date <= '{end_date}'
        ORDER BY date
        """
        
        print(f"Executing query: {query}")
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        print(df.head())

        if df.empty:
            print(f"No data found for {stock_symbol} between {extended_start_date} and {end_date}")
            print(f"Available data range: {min_date} to {max_date}")
        else:
            print(f"Retrieved {len(df)} rows of data, including historical data for feature engineering")
            
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
    try:
        # Get connection using Airflow's PostgresHook
        pg_hook = PostgresHook(postgres_conn_id='neon_db')
        conn = pg_hook.get_conn()
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
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
        
        # Insert data using batch execution with ON CONFLICT DO NOTHING
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
        ) ON CONFLICT (date, stock_symbol) DO NOTHING
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

def check_model_exists(stock_symbol: str) -> tuple:
    """
    Check if a model exists for the given stock and if it needs retraining.
    
    Args:
        stock_symbol: Stock symbol to check model for
        
    Returns:
        tuple: (exists: bool, needs_retraining: bool, model_path: str)
    """
    model_dir = "models"
    model_path = f"{model_dir}/{stock_symbol}_model.pkl"
    metadata_path = f"{model_dir}/{stock_symbol}_metadata.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        return False, True, model_path
        
    try:
        metadata = joblib.load(metadata_path)
        last_trained = metadata['last_trained']
        
        # Check if model is older than 30 days
        needs_retraining = (datetime.now() - last_trained).days > 30
        
        return True, needs_retraining, model_path
        
    except Exception as e:
        print(f"Error checking model metadata: {e}")
        return False, True, model_path

def train_model(stock_symbol: str, prediction_date_range: tuple = None) -> tuple:
    """
    Train a linear regression model on stock data.
    Pulls historical data from the database and trains on all data except the prediction date range.
    
    Args:
        stock_symbol: Stock symbol being modeled
        prediction_date_range: Tuple of (start_date, end_date) strings in YYYY-MM-DD format
                              This range will be excluded from training and used for predictions
        
    Returns:
        tuple: (model, scaler, metadata)
    """
    try:
        # Pull all historical data for the stock from the database
        pg_hook = PostgresHook(postgres_conn_id='neon_db')
        conn = pg_hook.get_conn()
        
        # Get min and max dates for the stock
        check_query = f"""
        SELECT 
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM ml_features 
        WHERE stock_symbol = '{stock_symbol}'
        """
        cursor = conn.cursor()
        cursor.execute(check_query)
        min_date, max_date = cursor.fetchone()
        
        # Get prediction date range
        if prediction_date_range:
            pred_start_date, pred_end_date = prediction_date_range
            print(f"Using prediction date range: {pred_start_date} to {pred_end_date}")
        else:
            # Default to last 90 days if no range specified
            pred_end_date = max_date
            pred_start_date = (pd.Timestamp(pred_end_date) - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
            print(f"No prediction date range specified. Using last 90 days: {pred_start_date} to {pred_end_date}")
        
        # Pull all data up to the prediction end date
        query = f"""
        SELECT * FROM ml_features 
        WHERE stock_symbol = '{stock_symbol}'
        AND date <= '{pred_end_date}'
        ORDER BY date
        """
        print(f"Pulling all historical data for {stock_symbol} up to {pred_end_date}")
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty:
            raise ValueError(f"No data found for {stock_symbol}")
            
        print(f"Retrieved {len(df)} rows of historical data for {stock_symbol}")
        
        # Convert date to datetime and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Split into training and validation sets
        # Training set: All data before prediction date range
        # Validation set: The prediction date range
        pred_start = pd.Timestamp(pred_start_date)
        mask = df['date'] < pred_start
        
        train_df = df[mask]
        val_df = df[~mask]
        
        print(f"Training data size: {len(train_df)} records, from {train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')}")
        print(f"Validation data size: {len(val_df)} records, from {val_df['date'].min().strftime('%Y-%m-%d')} to {val_df['date'].max().strftime('%Y-%m-%d')}")
        
        # Prepare features and target
        feature_cols = [
            'daily_sentiment_rollmean_7', 'daily_sentiment_lag_3', 'volume_rollmean_3',
            'volume_lag_2', 'adj_close_lag_3', 'adj_close_rollmean_3', 'volume_rollstd_14',
            'adj_close_lag_5', 'volume_lag_3', 'adj_close_rollmean_14', 'adj_close_lag_1',
            'adj_close_rollstd_14', 'adj_close_lag_2', 'daily_sentiment_lag_1',
            'fed_funds_rate', 'volume_lag_1', 'daily_sentiment_lag_2',
            'daily_sentiment_rollmean_14', 'sentiment_max', 'adj_close', 'article_count',
            'sentiment_min', 'adj_close_rollmean_7', 'atr_7d', 'daily_sentiment_rollmean_3',
            'sentiment_range', 'daily_sentiment_rollstd_14'
        ]
        
        X_train = train_df[feature_cols]
        y_train = train_df['target']
        
        X_val = val_df[feature_cols]
        y_val = val_df['target']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_val_scaled)
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        
        print(f"Model evaluation metrics:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Create metadata
        metadata = {
            'last_trained': datetime.now(),
            'feature_cols': feature_cols,
            'stock_symbol': stock_symbol,
            'model_type': 'LinearRegression',
            'train_size': len(X_train),
            'val_size': len(X_val),
            'train_period': (train_df['date'].min().strftime('%Y-%m-%d'), train_df['date'].max().strftime('%Y-%m-%d')),
            'val_period': (val_df['date'].min().strftime('%Y-%m-%d'), val_df['date'].max().strftime('%Y-%m-%d')),
            'prediction_date_range': prediction_date_range,
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
        }
        
        return model, scaler, metadata
        
    except Exception as e:
        print(f"Error training model: {e}")
        raise

def save_model(model, scaler, metadata, model_path: str):
    """
    Save model, scaler, and metadata to disk.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        metadata: Model metadata
        model_path: Path to save model to
    """
    try:
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and scaler
        joblib.dump({
            'model': model,
            'scaler': scaler
        }, model_path)
        
        # Save metadata
        metadata_path = model_path.replace('_model.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        print(f"Model and metadata saved to {model_path}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        raise

def load_model(model_path: str) -> tuple:
    """
    Load model, scaler, and metadata from disk.
    
    Args:
        model_path: Path to load model from
        
    Returns:
        tuple: (model, scaler, metadata)
    """
    try:
        # Load model and scaler
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Load metadata
        metadata_path = model_path.replace('_model.pkl', '_metadata.pkl')
        metadata = joblib.load(metadata_path)
        
        return model, scaler, metadata
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def make_predictions(df: pd.DataFrame, model, scaler, metadata: dict) -> pd.DataFrame:
    """
    Make predictions on new data.
    
    Args:
        df: DataFrame containing features
        model: Trained model
        scaler: Feature scaler
        metadata: Model metadata
        
    Returns:
        DataFrame with predictions
    """
    # Prepare features
    X = df[metadata['feature_cols']]
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'date': df['date'],
        'stock_symbol': df['stock_symbol'],
        'prediction': predictions,
        'prediction_date': datetime.now()
    })
    
    return results

def save_predictions_to_db(predictions: pd.DataFrame, table_name: str = "stock_predictions"):
    """
    Save predictions to database.
    
    Args:
        predictions: DataFrame containing predictions
        table_name: Name of the table to store predictions
    """
    try:
        pg_hook = PostgresHook(postgres_conn_id='neon_db')
        conn = pg_hook.get_conn()
        cursor = conn.cursor()
        
        # Check if table exists to avoid creating a duplicate type
        check_table_query = f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = '{table_name}'
        );
        """
        cursor.execute(check_table_query)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            # Create table only if it doesn't exist
            print(f"Creating new table {table_name}")
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                date DATE,
                stock_symbol TEXT,
                prediction DOUBLE PRECISION,
                prediction_date TIMESTAMP,
                PRIMARY KEY (date, stock_symbol)
            )
            """
            cursor.execute(create_table_query)
            conn.commit()
        else:
            print(f"Table {table_name} already exists, skipping creation")
        
        # Prepare records for batch insert
        records = [
            (
                row['date'],
                row['stock_symbol'],
                row['prediction'],
                row['prediction_date']
            )
            for _, row in predictions.iterrows()
        ]
        
        # Insert data using batch execution
        insert_query = f"""
        INSERT INTO {table_name} (
            date, stock_symbol, prediction, prediction_date
        ) VALUES (
            %s, %s, %s, %s
        )
        ON CONFLICT (date, stock_symbol) DO UPDATE
        SET prediction = EXCLUDED.prediction,
            prediction_date = EXCLUDED.prediction_date
        """
        
        execute_batch(cursor, insert_query, records)
        conn.commit()
        print(f"Inserted {len(records)} predictions into the database.")
        
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error inserting predictions: {e}")
        raise
    finally:
        if conn and not conn.closed:
            conn.close()

def run_ml_pipeline(stock_symbol: str):
    """
    Run the complete ML pipeline for a stock.
    
    Args:
        stock_symbol: Stock symbol to run pipeline for
    """
    try:
        # Pull stock data
        print(f"Pulling data for {stock_symbol}...")
        df = pull_stock_data(stock_symbol)
        
        # Engineer features
        print("Engineering features...")
        df_engineered = engineer_features(df)
        
        # Load features to DB
        print("Loading features to database...")
        load_features_to_db(df_engineered)
        
        # Check if model exists and needs retraining
        print("Checking model status...")
        model_exists, needs_retraining, model_path = check_model_exists(stock_symbol)
        
        if not model_exists or needs_retraining:
            print("Training new model...")
            model, scaler, metadata = train_model(stock_symbol)
            save_model(model, scaler, metadata, model_path)
        else:
            print("Loading existing model...")
            model, scaler, metadata = load_model(model_path)
        
        # Get last 90 days of data for prediction
        print("Making predictions on latest 90 days...")
        df_recent = df_engineered.iloc[-90:]
        predictions = make_predictions(df_recent, model, scaler, metadata)
        
        # Save predictions to DB
        print("Saving predictions to database...")
        save_predictions_to_db(predictions)
        
        print("ML pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in ML pipeline: {e}")
        raise 