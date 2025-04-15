import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sqlalchemy import create_engine
from typing import List
from airflow.providers.postgres.hooks.postgres import PostgresHook
import psycopg2
from psycopg2.extras import execute_batch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ------------------------------
# Extract articles
# ------------------------------
def extract_articles(top_stocks: List[str] = None, date_filter: str = 'all', start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Extracts articles related to the given list of stocks within a date range.
    If no stock list is provided, defaults to top 10 NASDAQ stocks.
    If start_date and end_date are provided, filters within that range.
    If date_filter is 'today', returns only today's articles.
    """
    if top_stocks is None:
        top_stocks = [
        "ADBE", "CMCSA", "QCOM", "GOOG", "PEP",
        "SBUX", "COST", "AMD", "INTC", "PYPL"
    ]
            
    ds = load_dataset("benstaf/FNSPID-filtered-nasdaq-100")
    df = ds["train"].to_pandas()
    
    # Convert to datetime and localize timezone to match input
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    # Filter by date range if provided
    if date_filter == 'today':
        today = pd.Timestamp.now().normalize()
        df = df[df['Date'] >= today]
    elif start_date and end_date:
        df = df[df['Date'] >= pd.to_datetime(start_date).tz_localize(None)]
        df = df[df['Date'] <= pd.to_datetime(end_date).tz_localize(None)]

    # Filter by selected symbols
    df = df[df['Stock_symbol'].isin(top_stocks)].copy()
    
    # Remove empty articles
    df = df[df['Article'].str.strip() != '']
    
    # Remove duplicates based on Article content and URL
    df = df.drop_duplicates(subset=['Article', 'Url'])

    return df

# ------------------------------
# Analyze sentiment
# ------------------------------
def analyze_finbert_sentiment(df, text_columns=None, model_name="ProsusAI/finbert"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    keep_cols = ['Date', 'Stock_symbol', 'Article_title', 'Article']
    if text_columns is None:
        text_columns = ['Lsa_summary', 'Luhn_summary', 'Textrank_summary', 'Lexrank_summary']
    df = df[keep_cols + text_columns].copy()

    label_to_score = {"positive": 1, "neutral": 0, "negative": -1}
    
    def get_average_sentiment(row):
        sentiments = []
        for col in text_columns:
            text = str(row[col])[:512]  # truncate to safe length
            try:
                result = sentiment_pipeline(text)[0]
                sentiment_label = result['label'].lower()
                confidence = result['score']
                score = label_to_score.get(sentiment_label, 0) * confidence
                sentiments.append(score)
            except Exception:
                sentiments.append(0)  # default to neutral on error
        return sum(sentiments) / len(sentiments) if sentiments else 0
    
    df['avg_sentiment'] = df.apply(get_average_sentiment, axis=1)
    
    return df

# ------------------------------
# Aggregate sentiment per day and stock
# ------------------------------
def aggregate_daily_sentiment(df, sentiment_column='avg_sentiment'):
    df = df.copy()
    
    # Ensure 'Date' is datetime if not already
    df['Date'] = pd.to_datetime(df['Date']).dt.date

    def compute_ratios(group):
        total = len(group)
        pos_ratio = (group[sentiment_column] > 0).sum() / total
        neg_ratio = (group[sentiment_column] < 0).sum() / total
        neu_ratio = (group[sentiment_column] == 0).sum() / total
        return pd.Series({
            'positive_ratio': pos_ratio,
            'negative_ratio': neg_ratio,
            'neutral_ratio': neu_ratio
        })

    # Aggregate basic stats
    df_stats = (
        df.groupby(['Stock_symbol', 'Date'])[sentiment_column]
          .agg([
              ('daily_sentiment', 'mean'),
              ('sentiment_std', 'std'), 
              ('article_count', 'count'),
              ('sentiment_median', 'median'),
              ('sentiment_min', 'min'),
              ('sentiment_max', 'max')
          ])
    )

    # Compute ratio features
    df_ratios = (
        df.groupby(['Stock_symbol', 'Date'])
          .apply(compute_ratios)
    )

    # Reset index once after merging to avoid duplicate columns
    df_daily = pd.merge(
        df_stats, 
        df_ratios,
        left_index=True,
        right_index=True
    ).reset_index()


    # Add sentiment range (max - min)
    df_daily['sentiment_range'] = df_daily['sentiment_max'] - df_daily['sentiment_min']

    return df_daily

# ------------------------------
# Airflow insert into PostgreSQL
# ------------------------------
def insert_article_sentiment(df: pd.DataFrame, conn_id: str) -> None:
    """
    Inserts daily article sentiment into the PostgreSQL database.
    Parameters:
        df (pd.DataFrame): DataFrame containing stock_symbol, date, daily_sentiment, article_count, sentiment_std,
                          positive_ratio, negative_ratio, neutral_ratio, sentiment_median, sentiment_min, sentiment_max,
                          sentiment_range.
        conn_id (str): Airflow connection ID for PostgreSQL.
    """
    hook = PostgresHook(postgres_conn_id=conn_id)
    conn = hook.get_conn()
    cur = conn.cursor()

    create_table_query = """
    CREATE TABLE IF NOT EXISTS daily_article_sentiment (
        stock_symbol TEXT NOT NULL,
        date DATE NOT NULL,
        daily_sentiment FLOAT,
        article_count INTEGER,
        sentiment_std FLOAT,
        positive_ratio FLOAT,
        negative_ratio FLOAT,
        neutral_ratio FLOAT,
        sentiment_median FLOAT,
        sentiment_min FLOAT,
        sentiment_max FLOAT,
        sentiment_range FLOAT,
        PRIMARY KEY (stock_symbol, date)
    );
    """
    insert_query = """
        INSERT INTO daily_article_sentiment (
            stock_symbol, date, daily_sentiment, article_count, sentiment_std,
            positive_ratio, negative_ratio, neutral_ratio, sentiment_median,
            sentiment_min, sentiment_max, sentiment_range
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (stock_symbol, date) DO NOTHING;
    """

    try:
        cur.execute(create_table_query)
        for _, row in df.iterrows():
            cur.execute(insert_query, (
                row['Stock_symbol'],
                row['Date'],
                row['daily_sentiment'],
                row['article_count'],
                row['sentiment_std'],
                row['positive_ratio'],
                row['negative_ratio'],
                row['neutral_ratio'],
                row['sentiment_median'],
                row['sentiment_min'],
                row['sentiment_max'],
                row['sentiment_range']
            ))
        conn.commit()
        print(f"Inserted {len(df)} article sentiment records into the database.")

    except Exception as e:
        conn.rollback()
        print(f"Error inserting data: {e}")

    finally:
        cur.close()
        conn.close()


# ------------------------------
# Manual insert into PostgreSQL
# ------------------------------
def insert_article_sentiment_manual(df: pd.DataFrame, db_params: dict) -> None:
    import psycopg2
    from psycopg2.extras import execute_batch

    create_table_query = """
    CREATE TABLE IF NOT EXISTS daily_article_sentiment (
        stock_symbol TEXT NOT NULL,
        date DATE NOT NULL,
        daily_sentiment FLOAT,
        article_count INTEGER,
        sentiment_std FLOAT,
        positive_ratio FLOAT,
        negative_ratio FLOAT,
        neutral_ratio FLOAT,
        sentiment_median FLOAT,
        sentiment_min FLOAT,
        sentiment_max FLOAT,
        sentiment_range FLOAT,
        PRIMARY KEY (stock_symbol, date)
    );
    """

    insert_query = """
    INSERT INTO daily_article_sentiment (
        stock_symbol, date, daily_sentiment, article_count, sentiment_std,
        positive_ratio, negative_ratio, neutral_ratio, sentiment_median,
        sentiment_min, sentiment_max, sentiment_range
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (stock_symbol, date) DO NOTHING;
    """

    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        cur.execute(create_table_query)

        records = [
            (
                row['Stock_symbol'],
                row['Date'],
                row['daily_sentiment'],
                row['article_count'],
                row['sentiment_std'],
                row['positive_ratio'],
                row['negative_ratio'],
                row['neutral_ratio'],
                row['sentiment_median'],
                row['sentiment_min'],
                row['sentiment_max'],
                row['sentiment_range']
            )
            for _, row in df.iterrows()
        ]

        execute_batch(cur, insert_query, records)
        conn.commit()
        print(f"Inserted {len(records)} article sentiment records into the database.")
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error inserting data: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()

# ------------------------------
# Script entry point (manual)
# ------------------------------
if __name__ == "__main__":
    tickers = ['AAPL']
    start_date = '2023-01-01'
    end_date = '2023-03-01'

    df_raw = extract_articles(top_stocks=tickers, start_date=start_date, end_date=end_date)
    df_scored = analyze_finbert_sentiment(df_raw)
    df_daily = aggregate_daily_sentiment(df_scored)

    print("\nSample aggregated sentiment data:")
    print(df_daily.head())

    # Using environment variables for database connection
    db_params = {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': int(os.getenv('DB_PORT')),
        'sslmode': os.getenv('DB_SSLMODE')
    }

    insert_article_sentiment_manual(df_daily, db_params)
    print("\nAll operations completed successfully!")