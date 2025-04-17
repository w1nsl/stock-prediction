from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import os
from dotenv import load_dotenv

from article_sentiment import extract_articles, analyze_sentiment, aggregate_daily_sentiment, insert_article_sentiment
from stock_price import download_stock_data, clean_stock_data, insert_stock_data
from us_economic_data import download_fred_data, insert_fred_data_manual
from merged_data import merge_all_data, insert_merged_data_to_db

load_dotenv()

STOCKS = [
    #"ADBE", "CMCSA", "QCOM", "GOOG", "PEP",
    #"SBUX", "COST", "AMD", "INTC", 
    "PYPL"
]

# Optional: Specify date range here. If None, will use execution date
START_DATE = "2023-03-15"  # Example: "2023-03-01"
END_DATE = "2023-03-31"    # Example: "2023-03-31"

def get_execution_date(**context):
    """Get execution date and ensure it's timezone-naive"""
    execution_date = context['execution_date']
    if execution_date.tzinfo:
        execution_date = execution_date.replace(tzinfo=None)
    return execution_date.strftime('%Y-%m-%d')
###################### EXTRACT TASKS ######################

def extract_stock_data(**context):
    pass
    '''try:
        if START_DATE and END_DATE:
            start_date = pd.Timestamp(START_DATE)
            end_date = pd.Timestamp(END_DATE)
            print(f"Using specified date range: {start_date.date()} to {end_date.date()}")
        else:
            end_date = pd.Timestamp(get_execution_date(**context))
            start_date = end_date - pd.DateOffset(days=7)
            print(f"No date range specified, using execution date: {end_date.date()}")
        
        print(f"Fetching stock data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        raw_data = download_stock_data(STOCKS, start_date, end_date)
        
        if not raw_data:
            print("No stock data found for any symbols. Using most recent available data...")
            current_date = pd.Timestamp.now()
            end_date = current_date
            start_date = end_date - pd.DateOffset(days=7)
            raw_data = download_stock_data(STOCKS, start_date, end_date)
        
        if raw_data:
            print(f"\nSuccessfully downloaded data for {len(raw_data)} out of {len(STOCKS)} stocks")
            return raw_data
        else:
            raise ValueError("Could not download data for any stocks")
            
    except Exception as e:
        error_msg = f"Failed to extract stock data: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)'''
    # Function disabled for compatibility

def extract_economic_data(**context):
    """Extract economic indicators from FRED"""
    try:
        # If dates are specified in the file, use those
        if START_DATE and END_DATE:
            start_date = pd.Timestamp(START_DATE)
            end_date = pd.Timestamp(END_DATE)
            print(f"Using specified date range: {start_date.date()} to {end_date.date()}")
        else:
            # Otherwise, use execution date
            end_date = pd.Timestamp(get_execution_date(**context))
            start_date = end_date - pd.DateOffset(days=7)
            print(f"No date range specified, using execution date: {end_date.date()}")
            
        print(f"Fetching economic data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        df = download_fred_data(start_date=start_date, end_date=end_date)
        return df
        
    except Exception as e:
        error_msg = f"Failed to extract economic data: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)

def extract_sentiment_data(**context):
    """Extract news articles for sentiment analysis"""
    try:
        if START_DATE and END_DATE:
            start_date = pd.Timestamp(START_DATE)
            end_date = pd.Timestamp(END_DATE)
            print(f"Using specified date range: {start_date.date()} to {end_date.date()}")
        else:
            end_date = pd.Timestamp(get_execution_date(**context))
            start_date = end_date - pd.DateOffset(days=7)
            print(f"No date range specified, using execution date: {end_date.date()}")
        
        print(f"Extracting articles for period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        raw_articles = extract_articles(
            STOCKS, 
            start_date=start_date.strftime('%Y-%m-%d'), 
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if raw_articles.empty:
            raise Exception("No articles were retrieved from the extraction process")
            
        print(f"Retrieved {len(raw_articles)} articles")
        
        data_dir = "/opt/airflow/data"
        os.makedirs(data_dir, exist_ok=True)
        
        output_path = f"{data_dir}/articles_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        raw_articles.to_parquet(output_path)
        print(f"Saved articles to {output_path}")
        
        return output_path
        
    except Exception as e:
        error_msg = f"Failed to extract articles: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)

###################### TRANSFORM TASKS ######################

def transform_stock_data(**context):
    pass
    """Clean and transform stock price data
    raw_data = context['task_instance'].xcom_pull(task_ids='extract_stock_data')
    cleaned_data = clean_stock_data(raw_data)
    return cleaned_data"""
    # Function disabled for compatibility
    
def transform_sentiment_data(**context):
    """Process articles and perform sentiment analysis"""
    try:
        import time
        start_time = time.time()
        print(f"Starting sentiment analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get the output path from the previous task
        ti = context['task_instance']
        articles_path = ti.xcom_pull(task_ids='extract_sentiment_data')
        print(f"Retrieved articles path: {articles_path}")
        
        # Read the articles from the parquet file
        read_start = time.time()
        articles_df = pd.read_parquet(articles_path)
        read_end = time.time()
        print(f"Read {len(articles_df)} articles from parquet file in {read_end - read_start:.2f} seconds")
        
        # Analyze sentiment
        sentiment_start = time.time()
        print(f"Starting sentiment analysis on {len(articles_df)} articles")
        sentiment_df = analyze_sentiment(articles_df)
        sentiment_end = time.time()
        print(f"Completed sentiment analysis in {sentiment_end - sentiment_start:.2f} seconds")
        
        # Save the sentiment results
        save_start = time.time()
        data_dir = "/opt/airflow/data"
        output_path = f"{data_dir}/sentiment_{os.path.basename(articles_path)}"
        sentiment_df.to_parquet(output_path)
        save_end = time.time()
        print(f"Saved sentiment analysis to {output_path} in {save_end - save_start:.2f} seconds")
        
        total_time = time.time() - start_time
        print(f"Total sentiment analysis process took {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        return output_path
        
    except Exception as e:
        error_msg = f"Failed to analyze sentiment: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)

def transform_merged_data(**context):
    """Merge stock, economic, and sentiment data"""
    try:
        # If dates are specified in the file, use those
        if START_DATE and END_DATE:
            start_date = pd.Timestamp(START_DATE)
            end_date = pd.Timestamp(END_DATE)
            print(f"Using specified date range: {start_date.date()} to {end_date.date()}")
        else:
            # Otherwise, use execution date
            end_date = pd.Timestamp(get_execution_date(**context))
            start_date = end_date - pd.DateOffset(days=7)
            print(f"No date range specified, using execution date: {end_date.date()}")
        
        # Ensure dates are timezone-naive
        start_date = start_date.tz_localize(None)
        end_date = end_date.tz_localize(None)
        
        merged_data = {}
        
        for ticker in STOCKS:
            merged = merge_all_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if not merged.empty:
                merged_data[ticker] = merged
        
        return merged_data
        
    except Exception as e:
        error_msg = f"Failed to merge data: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)

###################### LOAD TASKS ######################

def load_stock_data(**context):
    pass
    """Load transformed stock data into database
    try:
        import time
        start_time = time.time()
        print(f"Starting stock data load at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get the stock data from the previous task
        ti = context['task_instance']
        stock_data = ti.xcom_pull(task_ids='transform_stock_data')
        print(f"Retrieved stock data for {len(stock_data)} stocks")
        
        # Print sample of data to be inserted
        print("\nSample of data to be inserted:")
        for ticker, data in stock_data.items():
            print(f"\n{ticker} data:")
            print(data.head())
        
        # Insert into database
        db_start = time.time()
        insert_stock_data(stock_data, conn_id='neon_db')
        db_end = time.time()
        print(f"\nLoaded data into database in {db_end - db_start:.2f} seconds")
        
        total_time = time.time() - start_time
        print(f"Total database load process took {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        return "Stock data loaded successfully"
        
    except Exception as e:
        error_msg = f"Failed to load stock data to database: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)"""

def load_economic_data(**context):
    """Load economic data into database"""
    try:
        import time
        start_time = time.time()
        print(f"Starting economic data load at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get the economic data from the previous task
        ti = context['task_instance']
        economic_data = ti.xcom_pull(task_ids='extract_economic_data')
        print(f"Retrieved economic data with {len(economic_data)} records")
        
        # Print sample of data to be inserted
        print("\nSample of data to be inserted:")
        print(economic_data.head())
        
        # Insert into database
        db_start = time.time()
        insert_fred_data_manual(economic_data, conn_id='neon_db')
        db_end = time.time()
        print(f"\nLoaded data into database in {db_end - db_start:.2f} seconds")
        
        total_time = time.time() - start_time
        print(f"Total database load process took {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        return "Economic data loaded successfully"
        
    except Exception as e:
        error_msg = f"Failed to load economic data to database: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)

def load_sentiment_data(**context):
    """Load sentiment analysis results into database"""
    try:
        import time
        start_time = time.time()
        print(f"Starting database load at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get the sentiment data path from the previous task
        ti = context['task_instance']
        sentiment_path = ti.xcom_pull(task_ids='transform_sentiment_data')
        print(f"Retrieved sentiment data path: {sentiment_path}")
        
        # Read the sentiment data
        read_start = time.time()
        sentiment_df = pd.read_parquet(sentiment_path)
        read_end = time.time()
        print(f"Read {len(sentiment_df)} records from sentiment analysis in {read_end - read_start:.2f} seconds")
        
        # Print sample of data to be inserted
        print("\nSample of data to be inserted:")
        print(sentiment_df.head())
        
        # Aggregate daily sentiment
        agg_start = time.time()
        daily_sentiment = aggregate_daily_sentiment(sentiment_df)
        agg_end = time.time()
        print(f"\nAggregated into {len(daily_sentiment)} daily records in {agg_end - agg_start:.2f} seconds")
        print("\nSample of aggregated data:")
        print(daily_sentiment.head())
        
        # Insert into database
        db_start = time.time()
        insert_article_sentiment(daily_sentiment, conn_id='neon_db')
        db_end = time.time()
        print(f"\nLoaded data into database in {db_end - db_start:.2f} seconds")
        
        total_time = time.time() - start_time
        print(f"Total database load process took {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        return "Data loaded successfully"
        
    except Exception as e:
        error_msg = f"Failed to load sentiment data to database: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)

def load_merged_data(**context):
    """Load merged data into database"""
    try:
        import time
        start_time = time.time()
        print(f"Starting merged data load at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get the merged data from the previous task
        ti = context['task_instance']
        merged_data = ti.xcom_pull(task_ids='transform_merged_data')
        print(f"Retrieved merged data for {len(merged_data)} stocks")
        
        # Print sample of data to be inserted
        print("\nSample of data to be inserted:")
        for ticker, data in merged_data.items():
            print(f"\n{ticker} data:")
            print(data.head())
        
        # Insert into database
        db_start = time.time()
        for ticker, data in merged_data.items():
            insert_merged_data_to_db(data, table_name="merged_stocks_new")
        db_end = time.time()
        print(f"\nLoaded data into database in {db_end - db_start:.2f} seconds")
        
        total_time = time.time() - start_time
        print(f"Total database load process took {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        return "Merged data loaded successfully"
        
    except Exception as e:
        error_msg = f"Failed to load merged data to database: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)

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
    start_date=datetime(2025, 4, 17),
    catchup=False,
    tags=['stock', 'prediction']
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

# Update task dependencies
extract_stocks >> transform_stocks >> load_stocks
extract_economic  >> load_economic 
extract_sentiment >> transform_sentiment >> load_sentiment

[load_stocks, load_economic, load_sentiment] >> transform_merge >> load_merge