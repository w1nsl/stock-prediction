import yfinance as yf
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_batch
from datetime import datetime

def download_stock_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
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
            cleaned_df = df[['Close', 'Adj Close', 'Volume']].copy()
            cleaned_df = cleaned_df.dropna()
            cleaned_df = cleaned_df.sort_index()
            cleaned_data[ticker] = cleaned_df
        except Exception as e:
            print(f"Error cleaning {ticker}: {e}")
    return cleaned_data

def insert_stock_data(data_dict, db_params):

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS stock_data (
        ticker VARCHAR(10),
        date DATE,
        close NUMERIC,
        adj_close NUMERIC,
        volume BIGINT,
        PRIMARY KEY (ticker, date)
    """

    insert_sql = """
    INSERT INTO stock_data (ticker, date, close, adj_close, volume)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (ticker, date) DO NOTHING
    """

    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        cur.execute(create_table_sql)

        data_to_insert = []
        for ticker, df in data_dict.items():
            for date, row in df.iterrows():
                data_to_insert.append((
                    ticker,
                    date.to_pydatetime().date(),
                    row['Close'],
                    row['Adj Close'],
                    row['Volume']
                ))

        execute_batch(cur, insert_sql, data_to_insert)
        conn.commit()
        print(f"Successfully inserted data for {len(data_dict)} tickers")

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error inserting data: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":

    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    start_date = '2020-01-01'
    end_date = '2023-01-01'

    raw_data = download_stock_data(tickers, start_date, end_date)
    cleaned_data = clean_stock_data(raw_data)

    print("\nSample cleaned data:")
    for ticker, df in cleaned_data.items():
        print(f"\n{ticker} data (first 5 rows):")
        print(df.head())

    csv_filename = f'{ticker}_cleaned_data.csv'
    df.to_csv(csv_filename, index=True)
    print(f"Saved to {csv_filename}")

    # replace with actual credentials
    db_params = {
        'host': 'localhost',
        'database': 'stock_db',
        'user': 'postgres',
        'password': 'yourpassword',
        'port': '5432'  # default PostgreSQL port
    }

    insert_stock_data(cleaned_data, db_params)

    print("\nAll operations completed successfully!")
