import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
from article_sentiment import extract_articles, analyze_sentiment, aggregate_daily_sentiment, insert_article_sentiment
from airflow.providers.postgres.hooks.postgres import PostgresHook


# Define the list of stocks to track
STOCKS = [
    #"ADBE", "CMCSA", "QCOM", "GOOG", "PEP",
    # "SBUX", "COST", "AMD", "INTC", 
    "PYPL"
]

# Optional: Specify date range here. If None, will use today's date
START_DATE = "2023-03-01"
END_DATE = "2023-03-31"

def extract_article_data(**context):
    """Extract news articles for sentiment analysis"""
    try:
        # If dates are specified in the file, use those
        if START_DATE and END_DATE:
            start_date = pd.Timestamp(START_DATE)
            end_date = pd.Timestamp(END_DATE)
            print(f"Using specified date range: {start_date.date()} to {end_date.date()}")
        else:
            # Otherwise, use today's date
            today = pd.Timestamp.now().normalize()
            start_date = today
            end_date = today
            print(f"No date range specified, using today's date: {today.date()}")
        
        print(f"Extracting articles for period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Convert dates to string format for extract_articles
        raw_articles = extract_articles(
            STOCKS, 
            start_date=start_date.strftime('%Y-%m-%d'), 
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if raw_articles.empty:
            raise Exception("No articles were retrieved from the extraction process")
            
        print(f"Retrieved {len(raw_articles)} articles")
        
        # Create data directory if it doesn't exist
        data_dir = "/opt/airflow/data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Save the DataFrame to a file instead of returning it
        output_path = f"{data_dir}/articles_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        raw_articles.to_parquet(output_path)
        print(f"Saved articles to {output_path}")
        
        # Return just the path instead of the DataFrame
        return output_path
        
    except Exception as e:
        error_msg = f"Failed to extract articles: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)

def analyze_sentiment_data(**context):
    """Analyze sentiment of extracted articles"""
    try:
        import time
        start_time = time.time()
        print(f"Starting sentiment analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get the output path from the previous task
        ti = context['ti']
        articles_path = ti.xcom_pull(task_ids='extract_article_data')
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

def load_sentiment_to_db(**context):
    """Load sentiment data into the database"""
    try:
        import time
        start_time = time.time()
        print(f"Starting database load at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get the sentiment data path from the previous task
        ti = context['ti']
        sentiment_path = ti.xcom_pull(task_ids='analyze_sentiment_data')
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
    'test_article_extraction',
    default_args=default_args,
    description='Test DAG for article extraction only',
    schedule_interval='0 0 * * *',  # Run daily at midnight
    start_date=datetime(2025, 4, 16),
    catchup=False,
    tags=['test', 'articles']
)

# Extract Task
extract_articles_task = PythonOperator(
    task_id='extract_article_data',
    python_callable=extract_article_data,
    provide_context=True,
    dag=dag
)

# Sentiment Analysis Task
analyze_sentiment_task = PythonOperator(
    task_id='analyze_sentiment_data',
    python_callable=analyze_sentiment_data,
    provide_context=True,
    dag=dag
)

# Database Loading Task
load_db_task = PythonOperator(
    task_id='load_sentiment_to_db',
    python_callable=load_sentiment_to_db,
    provide_context=True,
    dag=dag
)

# Set task dependencies
extract_articles_task >> analyze_sentiment_task >> load_db_task 