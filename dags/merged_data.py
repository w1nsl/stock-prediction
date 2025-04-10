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

def merge_all_data(ticker: str, start_date: str, end_date: str, economic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge stock price, article sentiment, and economic data.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        economic_df: Pre-fetched economic data DataFrame
        
    Returns:
        DataFrame containing merged data
    """
    print(f"\nProcessing {ticker} from {start_date} to {end_date}")
    
    try:
        # Get stock data
        raw_stock_data = download_stock_data([ticker], start_date, end_date)
        if ticker not in raw_stock_data:
            raise ValueError(f"No stock data found for {ticker}")
            
        stock_df = clean_stock_data(raw_stock_data)[ticker].reset_index()
        stock_df['ticker'] = ticker
        
        # Get article sentiment data
        df_raw = extract_articles(top_stocks=[ticker], start_date=start_date, end_date=end_date)
        
        # Create empty sentiment DataFrame with all required columns if no articles found
        if df_raw.empty:
            print(f"No articles found for {ticker}, using default sentiment values")
            sentiment_df = pd.DataFrame({
                'Stock_symbol': [ticker],
                'Date': [pd.to_datetime(start_date).date()],
                'daily_sentiment': [0],
                'article_count': [0],
                'sentiment_std': [0],
                'positive_ratio': [0],
                'negative_ratio': [0],
                'neutral_ratio': [0],
                'sentiment_median': [0],
                'sentiment_min': [0],
                'sentiment_max': [0],
                'sentiment_range': [0]
            })
        else:
            df_scored = analyze_finbert_sentiment(df_raw)
            sentiment_df = aggregate_daily_sentiment(df_scored)
        
        # Rename columns for clarity and consistency
        stock_df = stock_df.rename(columns={
            'Date': 'date',
            'Open': 'open_price',
            'High': 'high_price',
            'Low': 'low_price',
            'Close': 'close_price',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        })
        
        # Ensure consistent column names in sentiment_df
        sentiment_df = sentiment_df.rename(columns={
            'Stock_symbol': 'stock_symbol',
            'Date': 'date'
        })
        
        # Convert date columns to datetime and ensure they're date objects
        stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
        
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
            'gdp', 'real_gdp', 'unemployment_rate', 'cpi', 'fed_funds_rate', 'sp500'
        ]
        for col in price_cols:
            final_df[col] = final_df.groupby("stock_symbol")[col].transform(lambda g: g.ffill())
        
        # Drop any remaining rows with missing values
        final_df = final_df.dropna(subset=price_cols)
        
        # Fill sentiment features
        sentiment_cols = [
            'daily_sentiment', 'article_count', 'sentiment_std',
            'positive_ratio', 'negative_ratio', 'neutral_ratio',
            'sentiment_median', 'sentiment_min', 'sentiment_max',
            'sentiment_range'
        ]
        for col in sentiment_cols:
            final_df[col] = final_df[col].fillna(0)
        
        return final_df
        
    except Exception as e:
        print(f"\nError processing {ticker}: {str(e)}")
        raise

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
        ON CONFLICT (date, stock_symbol) DO UPDATE SET
            open_price = EXCLUDED.open_price,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price,
            close_price = EXCLUDED.close_price,
            adj_close = EXCLUDED.adj_close,
            volume = EXCLUDED.volume,
            daily_sentiment = EXCLUDED.daily_sentiment,
            article_count = EXCLUDED.article_count,
            sentiment_std = EXCLUDED.sentiment_std,
            positive_ratio = EXCLUDED.positive_ratio,
            negative_ratio = EXCLUDED.negative_ratio,
            neutral_ratio = EXCLUDED.neutral_ratio,
            sentiment_median = EXCLUDED.sentiment_median,
            sentiment_min = EXCLUDED.sentiment_min,
            sentiment_max = EXCLUDED.sentiment_max,
            sentiment_range = EXCLUDED.sentiment_range,
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

def process_single_stock(ticker: str, start_date: str, end_date: str, db_params: dict, economic_df: pd.DataFrame) -> tuple:
    """
    Process a single stock and return the result.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        db_params: Database connection parameters
        economic_df: Pre-fetched economic data DataFrame
        
    Returns:
        tuple: (ticker, success, error_message)
    """
    try:
        merged_data = merge_all_data(ticker, start_date, end_date, economic_df)
        insert_merged_data_to_db(merged_data, db_params, table_name="merged_stocks_new")
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
    
    # Using environment variables for database connection
    db_params = {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': int(os.getenv('DB_PORT')),
        'sslmode': os.getenv('DB_SSLMODE')
    }
    
    # Fetch economic data once for all stocks
    print("Fetching economic data...")
    economic_df = download_fred_data(start_date=start_date, end_date=end_date)
    economic_df = economic_df.rename(columns={
        'GDP': 'gdp',
        'Real_GDP': 'real_gdp',
        'Unemployment_Rate': 'unemployment_rate',
        'CPI': 'cpi',
        'Fed_Funds_Rate': 'fed_funds_rate',
        'SP500': 'sp500'
    })
    economic_df['date'] = pd.to_datetime(economic_df['date']).dt.date
    
    # Process stocks in parallel
    print(f"\nProcessing {len(stocks)} stocks from {start_date} to {end_date}")
    
    # Create a partial function with fixed parameters
    process_func = partial(process_single_stock, 
                         start_date=start_date, 
                         end_date=end_date, 
                         db_params=db_params,
                         economic_df=economic_df)
    
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