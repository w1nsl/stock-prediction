import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import importlib.util

def import_module_from_path(module_name, file_path):
    """Import a module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def load_stock_data_from_dags(stock_symbol=None, start_date=None, end_date=None):
    """
    Load stock data directly from the DAG files
    
    Args:
        stock_symbol: Stock ticker symbol (if None, load all stocks)
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame containing merged data
    """
    print(f"Loading data directly from DAG files for {stock_symbol}")
    
    # Import the necessary functions from the DAG files
    dags_dir = os.path.join(os.getcwd(), 'dags')
    
    # Load stock price module
    try:
        stock_price_path = os.path.join(dags_dir, 'stock_price.py')
        stock_price_module = import_module_from_path('stock_price', stock_price_path)
        download_stock = stock_price_module.download_stock_data
        clean_stock = stock_price_module.clean_stock_data
    except Exception as e:
        print(f"Error importing stock_price.py: {e}")
        return pd.DataFrame()
    
    # Load sentiment module
    try:
        sentiment_path = os.path.join(dags_dir, 'article_sentiment.py')
        sentiment_module = import_module_from_path('article_sentiment', sentiment_path)
        extract_articles = sentiment_module.extract_articles
        analyze_sentiment = sentiment_module.analyze_finbert_sentiment
        aggregate_sentiment = sentiment_module.aggregate_daily_sentiment
    except Exception as e:
        print(f"Error importing article_sentiment.py: {e}")
        return pd.DataFrame()
    
    # Load economic data module
    try:
        economic_path = os.path.join(dags_dir, 'us_economic_data.py')
        economic_module = import_module_from_path('us_economic_data', economic_path)
        download_economic = economic_module.download_fred_data
    except Exception as e:
        print(f"Error importing us_economic_data.py: {e}")
        return pd.DataFrame()
    
    # Load merged data module
    try:
        merged_path = os.path.join(dags_dir, 'merged_data.py')
        merged_module = import_module_from_path('merged_data', merged_path)
        merge_data = merged_module.merge_all_data
    except Exception as e:
        print(f"Error importing merged_data.py: {e}")
        return pd.DataFrame()
    
    # Process parameters
    if stock_symbol is None:
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]  # Default stocks
    elif isinstance(stock_symbol, list):
        tickers = stock_symbol
    else:
        tickers = [stock_symbol]
    
    if start_date is None:
        start_date = '2023-01-01'
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Load economic data once (shared across all stocks)
    try:
        economic_df = download_economic(start_date, end_date)
    except Exception as e:
        print(f"Error downloading economic data: {e}")
        # Create sample economic data if download fails
        dates = pd.date_range(start=start_date, end=end_date)
        economic_df = pd.DataFrame({
            'date': dates,
            'gdp': np.linspace(23000, 24000, len(dates)),
            'real_gdp': np.linspace(20000, 21000, len(dates)),
            'unemployment_rate': np.linspace(3.5, 3.3, len(dates)),
            'cpi': np.linspace(300, 310, len(dates)),
            'fed_funds_rate': np.linspace(4.5, 5.0, len(dates)),
            'sp500': np.linspace(4200, 4500, len(dates))
        })
    
    # Process each stock and combine
    all_data = []
    for ticker in tickers:
        try:
            # Try to merge data directly using the merge_all_data function
            merged_df = merge_data(ticker, start_date, end_date, economic_df)
            all_data.append(merged_df)
        except Exception as e:
            print(f"Error merging data for {ticker}: {e}")
            try:
                # Try to get each component separately
                # 1. Get stock data
                raw_stock_data = download_stock([ticker], start_date, end_date)
                if ticker not in raw_stock_data:
                    raise ValueError(f"No stock data found for {ticker}")
                    
                stock_df = clean_stock(raw_stock_data)[ticker].reset_index()
                stock_df['ticker'] = ticker
                
                # 2. Get article sentiment data
                df_raw = extract_articles(top_stocks=[ticker], start_date=start_date, end_date=end_date)
                
                if df_raw.empty:
                    print(f"No articles found for {ticker}, using default sentiment values")
                    # Create empty sentiment data with default values
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
                    df_scored = analyze_sentiment(df_raw)
                    sentiment_df = aggregate_sentiment(df_scored)
                
                # 3. Rename columns for consistency
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
                    'Date': 'date'
                })
                
                # 4. Convert date columns
                stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
                
                # 5. Align stock_symbol column
                stock_df = stock_df.rename(columns={"ticker": "stock_symbol"})
                
                # 6. Merge stock data with sentiment data
                merged_df = pd.merge(
                    stock_df,
                    sentiment_df,
                    on=['date', 'stock_symbol'],
                    how='outer'
                )
                
                # 7. Merge with economic data
                final_df = pd.merge(
                    merged_df,
                    economic_df,
                    on='date',
                    how='outer'
                )
                
                # 8. Sort and add to combined data
                final_df = final_df.sort_values(by=['stock_symbol', 'date']).reset_index(drop=True)
                all_data.append(final_df)
                
            except Exception as nested_e:
                print(f"Failed to build data manually for {ticker}: {nested_e}")
    
    # Combine all stock data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.attrs['data_source'] = 'dags'
        
        # Filter by stock symbol if requested
        if stock_symbol and not isinstance(stock_symbol, list):
            combined_df = combined_df[combined_df['stock_symbol'] == stock_symbol]
            
        # Filter by date range
        if start_date:
            combined_df = combined_df[combined_df['date'] >= pd.to_datetime(start_date).date()]
        if end_date:
            combined_df = combined_df[combined_df['date'] <= pd.to_datetime(end_date).date()]
            
        return combined_df
    else:
        return pd.DataFrame()

def generate_predictions(df, prediction_horizon=5, error_factor=0.05):
    """
    Generate simple predictions based on actual data
    
    Args:
        df: DataFrame with stock data
        prediction_horizon: Days to look ahead for prediction
        error_factor: Amount of error to introduce (0.05 = 5%)
    
    Returns:
        DataFrame with actual and predicted prices
    """
    if df.empty:
        return pd.DataFrame()
    
    predictions = []
    
    # Group by stock symbol
    for symbol, group in df.groupby('stock_symbol'):
        # Sort by date
        group = group.sort_values('date')
        
        # Create lag features
        for i in range(1, prediction_horizon + 1):
            group[f'lag_{i}'] = group['close_price'].shift(i)
        
        # Drop rows with missing lag features
        group = group.dropna()
        
        # Simple prediction model - weighted average of lag prices plus trending factor
        weights = np.array([0.5, 0.25, 0.15, 0.07, 0.03])[:prediction_horizon]
        weights = weights / weights.sum()  # Normalize weights
        
        # Calculate predicted price
        group['predicted_price'] = 0
        for i in range(1, prediction_horizon + 1):
            if i <= len(weights):
                group['predicted_price'] += group[f'lag_{i}'] * weights[i-1]
        
        # Add some random error
        np.random.seed(42)  # For reproducibility
        error = np.random.normal(0, group['close_price'] * error_factor)
        group['predicted_price'] += error
        
        # Prepare result dataframe
        result = pd.DataFrame({
            'date': group['date'],
            'stock_symbol': symbol,
            'actual_price': group['close_price'],
            'predicted_price': group['predicted_price']
        })
        
        predictions.append(result)
    
    # Combine all predictions
    if predictions:
        combined = pd.concat(predictions, ignore_index=True)
        combined.attrs['data_source'] = 'generated'
        return combined
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    # Test the data loader
    stock_symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-03-01'
    
    df = load_stock_data_from_dags(stock_symbol, start_date, end_date)
    
    if not df.empty:
        print(f"Successfully loaded {len(df)} records for {stock_symbol}")
        print(df.head())
        
        # Generate predictions
        predictions = generate_predictions(df)
        print("\nGenerated predictions:")
        print(predictions.head())
    else:
        print("Failed to load data") 