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

def merge_all_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Merge stock price, article sentiment, and economic data from existing database tables.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame containing merged data
    """
    print(f"\nProcessing {ticker} from {start_date} to {end_date}")
    
    try:
        # Get the Postgres connection from Airflow
        pg_hook = PostgresHook(postgres_conn_id='neon_db')
        
        # Query to join all three tables
        query = """
        SELECT 
            sd.date,
            sd.ticker as stock_symbol,
            sd.open_price,
            sd.high_price,
            sd.low_price,
            sd.close_price,
            sd.adj_close,
            sd.volume,
            das.daily_sentiment,
            das.article_count,
            das.sentiment_std,
            das.positive_ratio,
            das.negative_ratio,
            das.neutral_ratio,
            das.sentiment_median,
            das.sentiment_min,
            das.sentiment_max,
            das.sentiment_range,
            ued.gdp,
            ued.real_gdp,
            ued.unemployment_rate,
            ued.cpi,
            ued.fed_funds_rate,
            ued.sp500
        FROM stock_data sd
        LEFT JOIN daily_article_sentiment das 
            ON sd.date = das.date AND sd.ticker = das.stock_symbol
        LEFT JOIN us_economic_data_daily ued 
            ON sd.date = ued.date
        WHERE sd.ticker = %s
            AND sd.date BETWEEN %s AND %s
        ORDER BY sd.date
        """
        
        # Execute the query
        df = pg_hook.get_pandas_df(query, parameters=(ticker, start_date, end_date))
        
        if df.empty:
            raise ValueError(f"No data found for {ticker} in the specified date range")
            
        # Ensure date is in the correct format
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Fill missing sentiment values with 0
        sentiment_cols = [
            'daily_sentiment', 'article_count', 'sentiment_std',
            'positive_ratio', 'negative_ratio', 'neutral_ratio',
            'sentiment_median', 'sentiment_min', 'sentiment_max',
            'sentiment_range'
        ]
        for col in sentiment_cols:
            df[col] = df[col].fillna(0)
            
        # Fill missing economic indicators using forward fill
        economic_cols = [
            'gdp', 'real_gdp', 'unemployment_rate', 
            'cpi', 'fed_funds_rate', 'sp500'
        ]
        for col in economic_cols:
            df[col] = df[col].ffill()
            
        return df
        
    except Exception as e:
        print(f"\nError processing {ticker}: {str(e)}")
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