import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_batch
import os
from dotenv import load_dotenv
from article_sentiment import extract_articles, analyze_sentiment, aggregate_daily_sentiment
from stock_price import download_stock_data, clean_stock_data
from us_economic_data import download_fred_data
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from airflow.hooks.postgres_hook import PostgresHook

# Load environment variables
load_dotenv()

def merge_all_data(stock_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Merge stock price, sentiment, and economic data for a given stock symbol and date range.
    Handles missing stock price data by forward filling from the previous day.
    
    Args:
        stock_symbol: Stock symbol to merge data for
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame containing merged data
    """
    try:
        # Get database connection
        pg_hook = PostgresHook(postgres_conn_id='neon_db')
        conn = pg_hook.get_conn()
        
        # Get stock price data
        stock_query = f"""
        SELECT * FROM stock_data 
        WHERE ticker = '{stock_symbol}'
        AND date >= '{start_date}'
        AND date <= '{end_date}'
        ORDER BY date
        """
        stock_df = pd.read_sql(stock_query, conn)
        
        # Create a complete date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        date_df = pd.DataFrame({'date': date_range})
        
        # If we have stock data, merge it with the complete date range
        if not stock_df.empty:
            # Convert date to datetime for merging
            stock_df['date'] = pd.to_datetime(stock_df['date'])
            
            # Merge with complete date range
            stock_df = date_df.merge(stock_df, on='date', how='left')
            
            # Forward fill missing stock data
            stock_cols = ['open_price', 'high_price', 'low_price', 'close_price', 
                         'adj_close', 'volume', 'stock_symbol']
            stock_df[stock_cols] = stock_df[stock_cols].ffill()
            
            # Backward fill stock_symbol for any remaining missing values
            stock_df['stock_symbol'] = stock_df['stock_symbol'].bfill()
        else:
            # If no stock data, create empty DataFrame with complete date range
            stock_df = date_df
            stock_df['stock_symbol'] = stock_symbol
            for col in ['open_price', 'high_price', 'low_price', 'close_price', 
                       'adj_close', 'volume']:
                stock_df[col] = None
        
        # Get sentiment data
        sentiment_query = f"""
        SELECT * FROM daily_article_sentiment 
        WHERE stock_symbol = '{stock_symbol}'
        AND date >= '{start_date}'
        AND date <= '{end_date}'
        ORDER BY date
        """
        sentiment_df = pd.read_sql(sentiment_query, conn)
        
        # Get economic data
        economic_query = f"""
        SELECT * FROM us_economic_data_daily 
        WHERE date >= '{start_date}'
        AND date <= '{end_date}'
        ORDER BY date
        """
        economic_df = pd.read_sql(economic_query, conn)
        
        # Close database connection
        conn.close()
        
        # Convert dates to datetime for merging
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        economic_df['date'] = pd.to_datetime(economic_df['date'])
        
        # Merge all data
        merged_df = stock_df.merge(sentiment_df, on=['date', 'stock_symbol'], how='left')
        merged_df = merged_df.merge(economic_df, on='date', how='left')
        
        # Fill missing values
        # Forward fill stock data
        stock_cols = ['open_price', 'high_price', 'low_price', 'close_price', 
                     'adj_close', 'volume']
        merged_df[stock_cols] = merged_df[stock_cols].ffill()
        
        # Forward fill economic data
        economic_cols = ['gdp', 'real_gdp', 'unemployment_rate', 'cpi', 
                        'fed_funds_rate', 'sp500']
        merged_df[economic_cols] = merged_df[economic_cols].ffill()
        
        # Fill sentiment data with zeros where missing
        sentiment_cols = ['daily_sentiment', 'article_count', 'sentiment_std', 
                         'positive_ratio', 'negative_ratio', 'neutral_ratio', 
                         'sentiment_median', 'sentiment_min', 'sentiment_max', 
                         'sentiment_range']
        merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(0)
        
        # Ensure stock_symbol is filled
        merged_df['stock_symbol'] = merged_df['stock_symbol'].fillna(stock_symbol)
        
        # Sort by date
        merged_df = merged_df.sort_values('date')
        
        return merged_df
        
    except Exception as e:
        print(f"Error merging data for {stock_symbol}: {str(e)}")
        raise

def insert_merged_data_to_db(df: pd.DataFrame, table_name: str = "merged_stock_data"):
    """
    Insert merged data into the database using Airflow connection.
    Skips entries that already exist in the database.
    
    Args:
        df: DataFrame containing merged data
        table_name: Name of the table to insert data into
    """
    try:
        # Get the Postgres connection from Airflow
        pg_hook = PostgresHook(postgres_conn_id='neon_db')
        conn = pg_hook.get_conn()
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            date DATE,
            stock_symbol TEXT,
            open_price NUMERIC,
            high_price NUMERIC,
            low_price NUMERIC,
            close_price NUMERIC,
            adj_close NUMERIC,
            volume BIGINT,
            daily_sentiment DOUBLE PRECISION,
            article_count INTEGER,
            sentiment_std DOUBLE PRECISION,
            positive_ratio DOUBLE PRECISION,
            negative_ratio DOUBLE PRECISION,
            neutral_ratio DOUBLE PRECISION,
            sentiment_median DOUBLE PRECISION,
            sentiment_min DOUBLE PRECISION,
            sentiment_max DOUBLE PRECISION,
            sentiment_range DOUBLE PRECISION,
            gdp DOUBLE PRECISION,
            real_gdp DOUBLE PRECISION,
            unemployment_rate DOUBLE PRECISION,
            cpi DOUBLE PRECISION,
            fed_funds_rate DOUBLE PRECISION,
            sp500 DOUBLE PRECISION,
            PRIMARY KEY (date, stock_symbol)
        )
        """
        cursor.execute(create_table_query)
        
        # Prepare records for batch insert
        records = [
            (
                row['date'],
                row['stock_symbol'],
                row['open_price'],
                row['high_price'],
                row['low_price'],
                row['close_price'],
                row['adj_close'],
                row['volume'],
                row['daily_sentiment'],
                row['article_count'],
                row['sentiment_std'],
                row['positive_ratio'],
                row['negative_ratio'],
                row['neutral_ratio'],
                row['sentiment_median'],
                row['sentiment_min'],
                row['sentiment_max'],
                row['sentiment_range'],
                row['gdp'],
                row['real_gdp'],
                row['unemployment_rate'],
                row['cpi'],
                row['fed_funds_rate'],
                row['sp500']
            )
            for _, row in df.iterrows()
        ]
        
        # Insert data using batch execution
        insert_query = f"""
        INSERT INTO {table_name} (
            date, stock_symbol, open_price, high_price, low_price, close_price,
            adj_close, volume, daily_sentiment, article_count, sentiment_std,
            positive_ratio, negative_ratio, neutral_ratio, sentiment_median,
            sentiment_min, sentiment_max, sentiment_range, gdp, real_gdp,
            unemployment_rate, cpi, fed_funds_rate, sp500
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (date, stock_symbol) DO NOTHING
        """
        
        execute_batch(cursor, insert_query, records)
        conn.commit()
        print(f"Inserted {len(records)} records into the database.")
        
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error inserting data: {e}")
        
    finally:
        if conn:
            cursor.close()
            conn.close()

def process_single_stock(ticker: str, start_date: str, end_date: str) -> tuple:
    """
    Process a single stock and return the result.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        tuple: (ticker, success, error_message)
    """
    try:
        merged_data = merge_all_data(ticker, start_date, end_date)
        insert_merged_data_to_db(merged_data, table_name="merged_stocks_new")
        print(f"Successfully processed {ticker}")
        return (ticker, True, None)
    except Exception as e:
        print(f"Failed to process {ticker}: {str(e)}")
        return (ticker, False, str(e))

if __name__ == "__main__":
    # List of stocks to process
    stocks = ["ADBE", "CMCSA", "QCOM", "GOOG", "PEP", "SBUX", "COST", "AMD", "INTC", "PYPL"]
    start_date = '2019-01-01'
    end_date = '2023-12-31'
    
    # Process stocks in parallel
    print(f"\nProcessing {len(stocks)} stocks from {start_date} to {end_date}")
    
    # Process stocks in parallel with progress bar
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.starmap(process_single_stock, [(ticker, start_date, end_date) for ticker in stocks]), 
                          total=len(stocks), 
                          desc="Processing stocks"))
    
    # Separate successful and failed stocks
    successful_stocks = [r[0] for r in results if r[1]]
    failed_stocks = [(r[0], r[2]) for r in results if not r[1]]
    
    # Print summary
    print("\nProcessing Summary:")
    print(f"Successfully processed: {len(successful_stocks)} stocks")
    if successful_stocks:
        print("Successful stocks:", ", ".join(successful_stocks))
    if failed_stocks:
        print(f"Failed to process: {len(failed_stocks)} stocks")
        print("Failed stocks with errors:")
        for ticker, error in failed_stocks:
            print(f"  - {ticker}: {error}")