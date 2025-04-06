import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import os
from dotenv import load_dotenv
from article_sentiment import extract_articles, analyze_finbert_sentiment, aggregate_daily_sentiment
from stock_price import download_stock_data, clean_stock_data
from us_economic_data import download_fred_data
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count
from functools import partial

# Load environment variables
load_dotenv()

def merge_all_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Merge stock price, article sentiment, and economic data.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame containing merged data
    """
    # Get stock data
    raw_stock_data = download_stock_data([ticker], start_date, end_date)
    stock_df = clean_stock_data(raw_stock_data)[ticker].reset_index()
    stock_df['ticker'] = ticker
    
    # Get article sentiment data
    df_raw = extract_articles(top_stocks=[ticker], start_date=start_date, end_date=end_date)
    df_scored = analyze_finbert_sentiment(df_raw)
    sentiment_df = aggregate_daily_sentiment(df_scored)
    
    # Get economic data
    economic_df = download_fred_data(start_date=start_date, end_date=end_date)
    
    # Rename columns for clarity
    stock_df = stock_df.rename(columns={
        'Date': 'date',
        'Open': 'open_price',
        'High': 'high_price',
        'Low': 'low_price',
        'Close': 'close_price',
        'Adj Close': 'adj_close',
        'Volume': 'volume'
    })
    
    sentiment_df = sentiment_df.rename(columns={
        'Stock_symbol': 'stock_symbol',
        'Date': 'date',
        'daily_sentiment': 'article_sentiment',
        'article_count': 'article_count',
        'sentiment_std': 'sentiment_std'
    })
    
    # Convert date columns to datetime
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    economic_df['date'] = pd.to_datetime(economic_df['date'])
    
    # Align stock_symbol column for merging
    stock_df = stock_df.rename(columns={"ticker": "stock_symbol"})
    
    # Merge stock data with sentiment data
    merged_df = pd.merge(
        stock_df,
        sentiment_df,
        on=['date', 'stock_symbol'],
        how='outer'
    )
    
    # Merge with economic data
    final_df = pd.merge(
        merged_df,
        economic_df,
        on='date',
        how='outer'
    )
    
    # Sort by date and stock_symbol
    final_df = final_df.sort_values(by=['stock_symbol', 'date']).reset_index(drop=True)
    
    # Fill missing stock prices and economic indicators
    price_cols = [
        'open_price', 'high_price', 'low_price', 'close_price', 'adj_close', 'volume',
        'GDP', 'Real_GDP', 'Unemployment_Rate', 'CPI', 'Fed_Funds_Rate', 'SP500'
    ]
    for col in price_cols:
        final_df[col] = final_df.groupby("stock_symbol")[col].transform(lambda g: g.ffill())
    
    # Drop any remaining rows with missing values
    final_df = final_df.dropna(subset=price_cols)
    
    # Fill sentiment and macro features
    final_df['article_sentiment'] = final_df['article_sentiment'].fillna(0)
    final_df['article_count'] = final_df['article_count'].fillna(0)
    final_df['sentiment_std'] = final_df['sentiment_std'].fillna(0)
    
    # Create binary hasSentiment flag
    final_df['hasSentiment'] = final_df['article_count'].apply(lambda x: 1 if x > 0 else 0)
    
    return final_df

def insert_merged_data_to_db(df: pd.DataFrame, db_params: dict, table_name: str = "merged_stock_data"):
    """
    Insert merged data into the database.
    
    Args:
        df: DataFrame containing merged data
        db_params: Dictionary containing database connection parameters
        table_name: Name of the table to insert data into
    """
    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            date DATE,
            stock_symbol VARCHAR(10),
            open_price FLOAT,
            high_price FLOAT,
            low_price FLOAT,
            close_price FLOAT,
            adj_close FLOAT,
            volume FLOAT,
            article_sentiment FLOAT,
            article_count INTEGER,
            sentiment_std FLOAT,
            hasSentiment INTEGER,
            gdp FLOAT,
            real_gdp FLOAT,
            unemployment_rate FLOAT,
            cpi FLOAT,
            fed_funds_rate FLOAT,
            sp500 FLOAT,
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
                row['article_sentiment'],
                row['article_count'],
                row['sentiment_std'],
                row['hasSentiment'],
                row['GDP'],
                row['Real_GDP'],
                row['Unemployment_Rate'],
                row['CPI'],
                row['Fed_Funds_Rate'],
                row['SP500']
            )
            for _, row in df.iterrows()
        ]
        
        # Insert data using batch execution
        insert_query = f"""
        INSERT INTO {table_name} (
            date, stock_symbol, open_price, high_price, low_price, close_price,
            adj_close, volume, article_sentiment, article_count, sentiment_std,
            hasSentiment, gdp, real_gdp, unemployment_rate, cpi, fed_funds_rate, sp500
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (date, stock_symbol) DO UPDATE SET
            open_price = EXCLUDED.open_price,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price,
            close_price = EXCLUDED.close_price,
            adj_close = EXCLUDED.adj_close,
            volume = EXCLUDED.volume,
            article_sentiment = EXCLUDED.article_sentiment,
            article_count = EXCLUDED.article_count,
            sentiment_std = EXCLUDED.sentiment_std,
            hasSentiment = EXCLUDED.hasSentiment,
            gdp = EXCLUDED.gdp,
            real_gdp = EXCLUDED.real_gdp,
            unemployment_rate = EXCLUDED.unemployment_rate,
            cpi = EXCLUDED.cpi,
            fed_funds_rate = EXCLUDED.fed_funds_rate,
            sp500 = EXCLUDED.sp500
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

def process_single_stock(ticker: str, start_date: str, end_date: str, db_params: dict) -> tuple:
    """
    Process a single stock and return the result.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        db_params: Database connection parameters
        
    Returns:
        tuple: (ticker, success, error_message)
    """
    try:
        merged_data = merge_all_data(ticker, start_date, end_date)
        insert_merged_data_to_db(merged_data, db_params, table_name="merged_stock_data_top")
        return (ticker, True, None)
    except Exception as e:
        return (ticker, False, str(e))

if __name__ == "__main__":
    # List of stocks to process
    stocks = ["ADBE", "CMCSA", "QCOM", "GOOG", "PEP", "SBUX", "COST", "AMD", "INTC", "PYPL"]
    start_date = '2020-01-01'
    end_date = '2024-12-31'
    
    # Using environment variables for database connection
    db_params = {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': int(os.getenv('DB_PORT')),
        'sslmode': os.getenv('DB_SSLMODE')
    }
    
    # Process stocks in parallel
    print(f"\nProcessing {len(stocks)} stocks from {start_date} to {end_date}")
    
    # Create a partial function with fixed parameters
    process_func = partial(process_single_stock, 
                         start_date=start_date, 
                         end_date=end_date, 
                         db_params=db_params)
    
    # Use number of CPU cores minus 1 to leave one core free for system processes
    num_processes = max(1, cpu_count() - 1)
    
    # Process stocks in parallel with progress bar
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_func, stocks), 
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