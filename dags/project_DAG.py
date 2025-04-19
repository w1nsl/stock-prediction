from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import os
from dotenv import load_dotenv

from article_sentiment import extract_articles, analyze_sentiment, aggregate_daily_sentiment, insert_article_sentiment
from stock_price import download_stock_data, clean_stock_data, insert_stock_data
from us_economic_data import download_fred_data, insert_fred_data_manual
from merged_data import merge_all_data, insert_merged_data_to_db
from ml_pipeline import pull_stock_data, engineer_features, load_features_to_db, check_model_exists, train_model, save_model, load_model, make_predictions, save_predictions_to_db

load_dotenv()

STOCKS = ["ADBE", "CMCSA", "QCOM", "GOOG", "PEP", "SBUX", "COST", "AMD", "INTC", "PYPL"]

# Specify date range here. If None, will use execution date
START_DATE = "2023-12-16"  
END_DATE = "2023-12-16"    

def get_execution_date(**context):
    """Get execution date and ensure it's timezone-naive"""
    execution_date = context['execution_date']
    if execution_date.tzinfo:
        execution_date = execution_date.replace(tzinfo=None)
    return execution_date.strftime('%Y-%m-%d')

@dag(
    dag_id='stock_prediction_pipeline',
    description='Stock prediction DAG with economic & sentiment data',
    schedule_interval='0 0 * * *',
    start_date=datetime(2025, 4, 17),
    catchup=False,
    tags=['stock', 'prediction'],
    default_args={
        'owner': 'airflow',
        'depends_on_past': False,
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    }
)
def stock_prediction_pipeline():

    ###################### EXTRACT TASKS ######################

    @task
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

    @task
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

    @task
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

    @task
    def transform_stock_data(raw_data=None, **context):
        pass
        """Clean and transform stock price data
        cleaned_data = clean_stock_data(raw_data)
        return cleaned_data"""
        # Function disabled for compatibility
        
    @task
    def transform_sentiment_data(articles_path, **context):
        """Process articles and perform sentiment analysis"""
        try:
            import time
            start_time = time.time()
            print(f"Starting sentiment analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
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

    @task
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

    @task
    def load_stock_data(stock_data=None, **context):
        pass
        """Load transformed stock data into database
        try:
            import time
            start_time = time.time()
            print(f"Starting stock data load at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
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

    @task
    def load_economic_data(economic_data, **context):
        """Load economic data into database"""
        try:
            import time
            start_time = time.time()
            print(f"Starting economic data load at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
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

    @task
    def load_sentiment_data(sentiment_path, **context):
        """Load sentiment analysis results into database"""
        try:
            import time
            start_time = time.time()
            print(f"Starting database load at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
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

    @task
    def load_merged_data(merged_data, **context):
        """Load merged data into database"""
        try:
            import time
            start_time = time.time()
            print(f"Starting merged data load at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
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

    @task
    def engineer_features_for_stocks(**context):
        """
        Pull stock data and engineer features for specified stocks and date range.
        Logs the engineered features for inspection.
        """
        try:
            import time
            start_time = time.time()
            print(f"Starting feature engineering at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Get date range from context or use defaults
            if START_DATE and END_DATE:
                start_date = pd.Timestamp(START_DATE)
                end_date = pd.Timestamp(END_DATE)
                print(f"Using specified date range: {start_date.date()} to {end_date.date()}")
            else:
                end_date = pd.Timestamp(get_execution_date(**context))
                start_date = end_date - pd.DateOffset(days=7)
                print(f"No date range specified, using execution date: {end_date.date()}")
            
            # Ensure dates are timezone-naive
            start_date = start_date.tz_localize(None)
            end_date = end_date.tz_localize(None)
            
            engineered_data = {}
            latest_date = None
            
            for ticker in STOCKS:
                print(f"\nProcessing {ticker}...")
                
                # Pull stock data
                print(f"Pulling data for {ticker} from {start_date.date()} to {end_date.date()}")
                df = pull_stock_data(ticker, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
                
                if df.empty:
                    print(f"No data found for {ticker}")
                    continue
                    
                # Engineer features
                print(f"Engineering features for {ticker}")
                df_engineered = engineer_features(df)
                
                # Log engineered features
                print("\nSample of engineered features:")
                print(df_engineered.head())
                
                engineered_data[ticker] = df_engineered
                
                # Save to database
                print(f"Loading features to database for {ticker}")
                load_features_to_db(df_engineered)
                
                # Update latest date
                ticker_latest_date = df_engineered['date'].max()
                if latest_date is None or ticker_latest_date > latest_date:
                    latest_date = ticker_latest_date
            
            # Calculate the prediction date range
            if latest_date is not None:
                prediction_start_date = latest_date - pd.DateOffset(days=90)
                print(f"Prediction date range: {prediction_start_date.date()} to {latest_date.date()}")
                # Convert Timestamps to strings before pushing to XCom
                context['task_instance'].xcom_push(key='prediction_date_range', value=(prediction_start_date.strftime('%Y-%m-%d'), latest_date.strftime('%Y-%m-%d')))
            
            total_time = time.time() - start_time
            print(f"\nTotal feature engineering process took {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            
            # Return stock symbols for downstream tasks that need to check models
            return STOCKS
            
        except Exception as e:
            error_msg = f"Failed to engineer features: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)

    @task
    def check_model_task_func(stock_symbol: str):
        """
        Function to check if a model exists for a given stock symbol and log the results.
        """
        import logging
        import os
        import sys
        import traceback
        
        logging.info(f"Starting model check for stock: {stock_symbol}")
        try:
            # Log paths being checked
            model_dir = "models"
            model_path = f"{model_dir}/{stock_symbol}_model.pkl"
            metadata_path = f"{model_dir}/{stock_symbol}_metadata.pkl"
            
            # Make sure the models directory exists
            os.makedirs(model_dir, exist_ok=True)
            
            logging.info(f"Checking model existence at: {model_path}")
            logging.info(f"Checking metadata existence at: {metadata_path}")
            
            # Check if model and metadata files exist
            model_exists = os.path.exists(model_path)
            metadata_exists = os.path.exists(metadata_path)
            
            logging.info(f"Model file exists: {model_exists}")
            logging.info(f"Metadata file exists: {metadata_exists}")
            
            # Get the result from check_model_exists function
            exists, needs_retraining, model_path = check_model_exists(stock_symbol)
            
            # Log detailed results
            logging.info(f"Final check result for {stock_symbol}:")
            logging.info(f"  Model exists: {exists}")
            logging.info(f"  Needs retraining: {needs_retraining}")
            logging.info(f"  Model path: {model_path}")
            
            return exists, needs_retraining, model_path
            
        except Exception as e:
            error_msg = f"Error checking model for {stock_symbol}: {str(e)}"
            stack_trace = traceback.format_exc()
            logging.error(error_msg)
            logging.error(f"Stack trace: {stack_trace}")
            
            # Return default values to allow the pipeline to continue
            model_dir = "models"
            model_path = f"{model_dir}/{stock_symbol}_model.pkl"
            return False, True, model_path

    @task
    def train_model_task(check_model_result, **context):
        """
        Train model for stocks that need training based on check_model results.
        Gets prediction date range from XCom pushed by engineer_features_for_stocks.
        
        Args:
            check_model_result: Tuple containing (exists, needs_retraining, model_path)
        """
        import logging
        
        # Unpack the check_model_result
        exists, needs_retraining, model_path = check_model_result
        
        # Extract stock symbol from model_path
        stock_symbol = os.path.basename(model_path).replace('_model.pkl', '')
        
        # Check if we need to train or retrain the model
        if not exists or needs_retraining:
            logging.info(f"Model for {stock_symbol} needs {'training' if not exists else 'retraining'}")
            
            # Get prediction date range from engineer_features_for_stocks task
            ti = context['task_instance']
            prediction_date_range = ti.xcom_pull(task_ids='engineer_features_for_stocks', key='prediction_date_range')
            logging.info(f"Retrieved prediction date range: {prediction_date_range}")
            
            # Train the model
            logging.info(f"Training model for {stock_symbol}...")
            model, scaler, metadata = train_model(stock_symbol, prediction_date_range)
            
            # Save the model
            logging.info(f"Saving model for {stock_symbol}...")
            save_model(model, scaler, metadata, model_path)
            
            # Store model evaluation metrics in database
            logging.info(f"Storing model evaluation metrics in database...")
            metrics = metadata['metrics']
            training_date = metadata['last_trained']
            
            # Get database connection
            pg_hook = PostgresHook(postgres_conn_id='neon_db')
            conn = pg_hook.get_conn()
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            create_table_query = """
            CREATE TABLE IF NOT EXISTS model_evaluations (
                id SERIAL PRIMARY KEY,
                stock_symbol TEXT NOT NULL,
                training_date TIMESTAMP NOT NULL,
                mae DOUBLE PRECISION,
                rmse DOUBLE PRECISION,
                r2 DOUBLE PRECISION,
                model_path TEXT,
                action TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_model_eval_stock ON model_evaluations(stock_symbol);
            CREATE INDEX IF NOT EXISTS idx_model_eval_date ON model_evaluations(training_date);
            """
            cursor.execute(create_table_query)
            conn.commit()
            
            # Insert metrics
            insert_query = """
            INSERT INTO model_evaluations (stock_symbol, training_date, mae, rmse, r2, model_path, action)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            action = 'trained' if not exists else 'retrained'
            cursor.execute(insert_query, (
                stock_symbol,
                training_date,
                metrics['mae'],
                metrics['rmse'],
                metrics['r2'],
                model_path,
                action
            ))
            conn.commit()
            cursor.close()
            conn.close()
            
            return {
                'stock_symbol': stock_symbol,
                'action': 'trained' if not exists else 'retrained',
                'model_path': model_path,
                'metrics': metadata['metrics']
            }
        else:
            logging.info(f"Model for {stock_symbol} is up to date. No training needed.")
            return {
                'stock_symbol': stock_symbol,
                'action': 'none',
                'model_path': model_path
            }

    @task
    def make_predictions_task(train_model_result, **context):
        """
        Load model and make predictions for the specified date range.
        
        Args:
            train_model_result: Result from train_model_task containing stock_symbol and model_path
        """
        import logging
        
        # Extract info from train_model_result
        stock_symbol = train_model_result['stock_symbol']
        model_path = train_model_result['model_path']
        
        # Get prediction date range from engineer_features_for_stocks task
        ti = context['task_instance']
        prediction_date_range = ti.xcom_pull(task_ids='engineer_features_for_stocks', key='prediction_date_range')
        pred_start_date, pred_end_date = prediction_date_range
        
        logging.info(f"Making predictions for {stock_symbol} from {pred_start_date} to {pred_end_date}")
        
        try:
            # Load the model
            logging.info(f"Loading model from {model_path}")
            model, scaler, metadata = load_model(model_path)
            
            # Pull data for the prediction date range
            logging.info(f"Pulling feature data for prediction date range")
            pg_hook = PostgresHook(postgres_conn_id='neon_db')
            conn = pg_hook.get_conn()
            
            # Query to get features for the prediction date range
            query = f"""
            SELECT * FROM ml_features 
            WHERE stock_symbol = '{stock_symbol}'
            AND date BETWEEN '{pred_start_date}' AND '{pred_end_date}'
            ORDER BY date
            """
            
            df_pred = pd.read_sql(query, conn)
            conn.close()
            
            if df_pred.empty:
                logging.warning(f"No data found for {stock_symbol} in the prediction range")
                return {
                    'stock_symbol': stock_symbol,
                    'predictions_made': 0,
                    'status': 'no_data'
                }
            
            logging.info(f"Retrieved {len(df_pred)} rows for prediction")
            
            # Make predictions
            logging.info(f"Making predictions...")
            predictions = make_predictions(df_pred, model, scaler, metadata)
            
            # Save predictions to DB
            logging.info(f"Saving {len(predictions)} predictions to database")
            save_predictions_to_db(predictions)
            
            return {
                'stock_symbol': stock_symbol,
                'predictions_made': len(predictions),
                'date_range': prediction_date_range,
                'status': 'success'
            }
            
        except Exception as e:
            logging.error(f"Error making predictions for {stock_symbol}: {str(e)}")
            return {
                'stock_symbol': stock_symbol,
                'status': 'error',
                'error': str(e)
            }

    # Extract
    stock_data = extract_stock_data()
    economic_data = extract_economic_data()
    articles_path = extract_sentiment_data()
    
    # Transform
    transformed_stock_data = transform_stock_data(stock_data)
    sentiment_data_path = transform_sentiment_data(articles_path)
    
    # These tasks can run after all extraction tasks are complete
    loaded_economic = load_economic_data(economic_data)
    loaded_sentiment = load_sentiment_data(sentiment_data_path)
    loaded_stocks = load_stock_data(transformed_stock_data)
    
    # Merge data after all data is loaded
    merged_data = transform_merged_data()
    merged_data.set_upstream([loaded_stocks, loaded_economic, loaded_sentiment])
    
    # Load merged data
    loaded_merged = load_merged_data(merged_data)
    
    # Engineer features after merged data is loaded
    stocks_for_model_check = engineer_features_for_stocks()
    stocks_for_model_check.set_upstream(loaded_merged)

    # Check models after feature engineering
    check_model_results = check_model_task_func.expand(stock_symbol=STOCKS)
    check_model_results.set_upstream(stocks_for_model_check)
    
    # Train models that need training
    train_model_results = train_model_task.expand(check_model_result=check_model_results)
    
    # Make predictions using trained models
    prediction_results = make_predictions_task.expand(train_model_result=train_model_results)

# Create the DAG
stock_prediction_dag = stock_prediction_pipeline()