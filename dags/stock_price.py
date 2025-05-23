import yfinance as yf
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_batch
from datetime import datetime
import os
from dotenv import load_dotenv
from airflow.providers.postgres.hooks.postgres import PostgresHook

# Load environment variables
load_dotenv()

def download_stock_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
            if not df.empty: #if can dl data then add
                data[ticker] = df
            else:
                print(f"No data found for {ticker}")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    return data

def clean_stock_data(data_dict):
    cleaned_data = {}
    for ticker, df in data_dict.items():
        try:
            # Reset the index to make Date a column
            df = df.reset_index()
            # Drop the Ticker level from columns
            df.columns = df.columns.droplevel(1)
            # Keep all columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            # Set Date as index again
            df = df.set_index('Date')
            cleaned_data[ticker] = df
        except Exception as e:
            print(f"Error cleaning {ticker}: {e}")
    return cleaned_data

def insert_stock_data(data_dict, conn_id):
    """Insert stock data into PostgreSQL database using Airflow connection"""
    create_table_sql = """
    DROP TABLE IF EXISTS stock_data;
    CREATE TABLE stock_data (
        ticker VARCHAR(10),
        date DATE,
        open_price NUMERIC,
        high_price NUMERIC,
        low_price NUMERIC,
        close_price NUMERIC,
        adj_close NUMERIC,
        volume BIGINT,
        PRIMARY KEY (ticker, date)
    );
    """

    insert_sql = """
    INSERT INTO stock_data (ticker, date, open_price, high_price, low_price, close_price, adj_close, volume)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (ticker, date) DO NOTHING
    """

    try:
        # Get connection using Airflow hook
        hook = PostgresHook(postgres_conn_id=conn_id)
        conn = hook.get_conn()
        cur = conn.cursor()
        
        # Create table
        cur.execute(create_table_sql)

        # Prepare data for insertion
        data_to_insert = []
        for ticker, df in data_dict.items():
            for date, row in df.iterrows():
                data_to_insert.append((
                    ticker,
                    date.to_pydatetime().date(),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    float(row['Adj Close']),
                    int(row['Volume'])
                ))

        # Insert data
        execute_batch(cur, insert_sql, data_to_insert)
        conn.commit()
        print(f"Successfully inserted data for {len(data_dict)} tickers")

    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        print(f"Error inserting data: {e}")
        raise
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    tickers = [
        "ADBE", "CMCSA", "QCOM", "GOOG", "PEP",
        "SBUX", "COST", "AMD", "INTC", "PYPL"
    ]
    start_date = '2019-01-01'
    end_date = '2023-12-31'

    raw_data = download_stock_data(tickers, start_date, end_date)
    cleaned_data = clean_stock_data(raw_data)

    print("\nSample cleaned data:")
    for ticker, df in cleaned_data.items():
        print(f"\n{ticker} data (first 5 rows):")
        print(df.head())
        
        # Save CSV for each ticker
        ''' csv_filename = f'{ticker}_cleaned_data.csv'
        df.to_csv(csv_filename, index=True)
        print(f"Saved to {csv_filename}")'''

    # Use Airflow connection ID
    insert_stock_data(cleaned_data, 'neon_db')
    print("\nAll operations completed successfully!")
