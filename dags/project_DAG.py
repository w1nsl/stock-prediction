from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import os
from dotenv import load_dotenv
import yfinance as yf

from article_sentiment import extract_articles, analyze_finbert_sentiment, aggregate_daily_sentiment, insert_article_sentiment
from stock_price import download_stock_data, clean_stock_data, insert_stock_data
from us_economic_data import download_fred_data, insert_fred_data_manual
from merged_data import merge_all_data, insert_merged_data_to_db
from Connection import Connection

load_dotenv()

STOCKS = [
    "ADBE", "CMCSA", "QCOM", "GOOG", "PEP",
    "SBUX", "COST", "AMD", "INTC", "PYPL"
]

def get_execution_date(**context):
    """Get execution date and ensure it's timezone-naive"""
    execution_date = context['execution_date']
    if execution_date.tzinfo:
        execution_date = execution_date.replace(tzinfo=None)
    return execution_date.strftime('%Y-%m-%d')

# Database connection parameters
DB_PARAMS = {
    'host': os.getenv('DB_HOST', 'postgres'),
    'database': os.getenv('DB_NAME', 'airflow'),
    'user': os.getenv('DB_USER', 'airflow'),
    'password': os.getenv('DB_PASSWORD', 'airflow'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'sslmode': os.getenv('DB_SSLMODE', 'prefer')
}

###################### EXTRACT TASKS ######################

def extract_stock_data(**context):
    """Extract stock price data from Yahoo Finance"""
    requested_end_date = pd.Timestamp(get_execution_date(**context))
    current_date = pd.Timestamp.now()
    
    # If requested date is in the future, use current date instead
    end_date = min(requested_end_date, current_date)
    start_date = end_date - pd.DateOffset(days=7)
    
    print(f"Fetching stock data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    raw_data = download_stock_data(STOCKS, start_date, end_date)
    
    if not raw_data:
        print("No stock data found for any symbols. Using most recent available data...")
        # Try getting most recent data
        end_date = current_date
        start_date = end_date - pd.DateOffset(days=7)
        raw_data = download_stock_data(STOCKS, start_date, end_date)
    
    return raw_data
    
    if raw_data:
        print(f"\nSuccessfully downloaded data for {len(raw_data)} out of {len(STOCKS)} stocks")
        return raw_data
    else:
        raise ValueError("Could not download data for any stocks")

def extract_economic_data(**context):
    """Extract economic indicators from FRED"""
    end_date = pd.Timestamp(get_execution_date(**context))
    start_date = end_date - pd.DateOffset(days=7)
    df = download_fred_data(start_date=start_date, end_date=end_date)
    return df

def extract_sentiment_data(**context):
    """Extract news articles for sentiment analysis"""
    end_date = pd.Timestamp(get_execution_date(**context))
    start_date = end_date - pd.DateOffset(days=7)
    # Convert dates to string format for extract_articles
    raw_articles = extract_articles(
        STOCKS, 
        start_date=start_date.strftime('%Y-%m-%d'), 
        end_date=end_date.strftime('%Y-%m-%d')
    )
    return raw_articles

###################### TRANSFORM TASKS ######################

def transform_stock_data(**context):
    """Clean and transform stock price data"""
    raw_data = context['task_instance'].xcom_pull(task_ids='extract_stock_data')
    cleaned_data = clean_stock_data(raw_data)
    return cleaned_data

def transform_sentiment_data(**context):
    """Process articles and perform sentiment analysis"""
    raw_articles = context['task_instance'].xcom_pull(task_ids='extract_sentiment_data')
    if not raw_articles.empty:
        # Ensure Date column is datetime type before sentiment analysis
        raw_articles['Date'] = pd.to_datetime(raw_articles['Date']).dt.tz_localize(None)
        scored_articles = analyze_finbert_sentiment(raw_articles)
        daily_sentiment = aggregate_daily_sentiment(scored_articles)
        return daily_sentiment
    return pd.DataFrame()  # Return empty DataFrame if no articles found

def transform_merged_data(**context):
    """Merge stock, economic, and sentiment data"""
    end_date = pd.Timestamp(get_execution_date(**context))
    start_date = end_date - pd.DateOffset(days=7)
    
    # Ensure dates are timezone-naive
    start_date = start_date.tz_localize(None)
    end_date = end_date.tz_localize(None)
    
    econ_df = context['task_instance'].xcom_pull(task_ids='extract_economic_data')
    merged_data = {}
    
    for ticker in STOCKS:
        merged = merge_all_data(ticker, start_date, end_date, econ_df)
        if not merged.empty:
            merged_data[ticker] = merged
    
    return merged_data

###################### LOAD TASKS ######################

def load_stock_data(**context):
    """Load transformed stock data into database"""
    cleaned_data = context['task_instance'].xcom_pull(task_ids='transform_stock_data')
    insert_stock_data(cleaned_data, DB_PARAMS)
    return "Stock data loaded"

def load_economic_data(**context):
    """Load economic data into database"""
    economic_data = context['task_instance'].xcom_pull(task_ids='extract_economic_data')
    insert_fred_data_manual(economic_data, DB_PARAMS)
    return "Economic data loaded"

def load_sentiment_data(**context):
    """Load sentiment analysis results into database"""
    sentiment_data = context['task_instance'].xcom_pull(task_ids='transform_sentiment_data')
    insert_article_sentiment(sentiment_data, 'postgres_default')
    return "Sentiment data loaded"

def load_merged_data(**context):
    """Load merged data into database"""
    merged_data = context['task_instance'].xcom_pull(task_ids='transform_merged_data')
    for ticker, data in merged_data.items():
        insert_merged_data_to_db(data, DB_PARAMS)
    return "Merged data loaded"

def init_database(**context):
    """Initialize the database tables"""
    connection = Connection()
    return connection.initialise_db(connection, 'postgres_default')

# Default args and DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stock_prediction_pipeline',
    default_args=default_args,
    description='Stock prediction DAG with economic & sentiment data',
    schedule_interval='0 0 * * *',
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['stock', 'prediction']
)

# Initialize Database
init_db = PythonOperator(
    task_id='init_database',
    python_callable=init_database,
    provide_context=True,
    dag=dag
)

# Extract Tasks
extract_stocks = PythonOperator(
    task_id='extract_stock_data',
    python_callable=extract_stock_data,
    provide_context=True,
    dag=dag
)

extract_economic = PythonOperator(
    task_id='extract_economic_data',
    python_callable=extract_economic_data,
    provide_context=True,
    dag=dag
)

extract_sentiment = PythonOperator(
    task_id='extract_sentiment_data',
    python_callable=extract_sentiment_data,
    provide_context=True,
    dag=dag
)

# Transform Tasks
transform_stocks = PythonOperator(
    task_id='transform_stock_data',
    python_callable=transform_stock_data,
    provide_context=True,
    dag=dag
)

transform_sentiment = PythonOperator(
    task_id='transform_sentiment_data',
    python_callable=transform_sentiment_data,
    provide_context=True,
    dag=dag
)

transform_merge = PythonOperator(
    task_id='transform_merged_data',
    python_callable=transform_merged_data,
    provide_context=True,
    dag=dag
)

# Load Tasks
load_stocks = PythonOperator(
    task_id='load_stock_data',
    python_callable=load_stock_data,
    provide_context=True,
    dag=dag
)

load_economic = PythonOperator(
    task_id='load_economic_data',
    python_callable=load_economic_data,
    provide_context=True,
    dag=dag
)

load_sentiment = PythonOperator(
    task_id='load_sentiment_data',
    python_callable=load_sentiment_data,
    provide_context=True,
    dag=dag
)

load_merge = PythonOperator(
    task_id='load_merged_data',
    python_callable=load_merged_data,
    provide_context=True,
    dag=dag
)

# Set up Dependencies
init_db >> [extract_stocks, extract_economic, extract_sentiment]

# Extract -> Transform
extract_stocks >> transform_stocks
extract_sentiment >> transform_sentiment
[extract_economic, transform_stocks, transform_sentiment] >> transform_merge

# Transform -> Load
transform_stocks >> load_stocks
extract_economic >> load_economic
transform_sentiment >> load_sentiment
transform_merge >> load_merge