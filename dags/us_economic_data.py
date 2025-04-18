import pandas as pd
from fredapi import Fred
from typing import List, Dict
import psycopg2
from psycopg2.extras import execute_batch
from psycopg2 import Error as PgError
import os
from dotenv import load_dotenv
from airflow.providers.postgres.hooks.postgres import PostgresHook

# Load environment variables
load_dotenv()

FRED_API_KEY = '5634d0081e84d747c4413186eb2c19cb'
fred = Fred(api_key=FRED_API_KEY)


def download_fred_data(start_date=None, end_date=None) -> pd.DataFrame:
    """Download economic data from FRED API and convert to daily frequency."""
    # Set end_date to today if not provided
    if end_date is None:
        end_date = pd.Timestamp.now().normalize()
    elif isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)

    # Set start_date to 5 years before end_date if not provided
    if start_date is None:
        start_date = end_date - pd.DateOffset(years=5)
    elif isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)

    # Extend start date by 1 year to ensure we have data to forward fill
    query_start = start_date - pd.DateOffset(years=1)
    
    print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"(Using {query_start.strftime('%Y-%m-%d')} as query start to ensure data availability)")

    series_dict = {
        'GDP': {'code': 'GDP', 'freq': 'Q'},           # Quarterly
        'Real_GDP': {'code': 'GDPC1', 'freq': 'Q'},    # Quarterly
        'Unemployment_Rate': {'code': 'UNRATE', 'freq': 'M'},  # Monthly
        'CPI': {'code': 'CPIAUCSL', 'freq': 'M'},      # Monthly
        'Fed_Funds_Rate': {'code': 'EFFR', 'freq': 'D'},  # Daily
        'SP500': {'code': 'SP500', 'freq': 'D'}        # Daily
    }

    def get_daily_data(series_info, name):
        try:
            # Get data with explicit date strings, starting earlier for fill
            start_str = query_start.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            data = fred.get_series(
                series_info['code'],
                observation_start=start_str,
                observation_end=end_str
            )
            
            if data is None or data.empty:
                print(f"Warning: No data retrieved for {name} ({series_info['code']})")
                # Create empty DataFrame with complete date range
                empty_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='D'))
                empty_df[name] = None
                return empty_df
            
            print(f"Retrieved {len(data)} observations for {name}")
            
            # Convert to DataFrame and handle dates
            df = data.to_frame(name=name)
            df.index = pd.to_datetime(df.index)
            
            # Create complete date range (including weekends)
            daily_index = pd.date_range(start=query_start, end=end_date, freq='D')
            
            # Handle different frequencies
            if series_info['freq'] == 'Q':
                # For quarterly data, resample to end of quarter first
                df = df.resample('QE').last()
                df = df.reindex(daily_index)
                # Forward fill all gaps for quarterly data (no limit)
                df[name] = df[name].ffill()
            elif series_info['freq'] == 'M':
                # For monthly data, resample to end of month first
                df = df.resample('ME').last()
                df = df.reindex(daily_index)
                # Forward fill all gaps for monthly data (no limit)
                df[name] = df[name].ffill()
            else:
                # For daily data, reindex to all days
                df = df.reindex(daily_index)
                # Forward fill all gaps for daily data (no limit)
                df[name] = df[name].ffill()
            
            # Filter to requested date range
            df = df[df.index >= start_date]
            df = df[df.index <= end_date]
            
            # Print data availability for debugging
            available = df[name].notna().sum()
            total = len(df)
            if total > 0:  # Avoid division by zero
                print(f"{name}: {available}/{total} data points available ({(available/total)*100:.1f}%)")
            else:
                print(f"{name}: No data points in date range")
            
            return df
            
        except Exception as e:
            print(f"Error downloading {name} data: {e}")
            # Return empty DataFrame with the correct index and column
            empty_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='D'))
            empty_df[name] = None
            return empty_df

    # Download and combine all data
    all_dfs = []
    for name, info in series_dict.items():
        df = get_daily_data(info, name)
        all_dfs.append(df)
        
    # If start_date and end_date are the same, ensure we have at least that date
    if start_date == end_date:
        # Make sure we have the exact date even if it's not a business day
        exact_date_index = pd.DatetimeIndex([start_date])
        for i, df in enumerate(all_dfs):
            name = df.columns[0]
            # Get the most recent value before this date
            if not df.empty:
                prev_values = df[df.index < start_date]
                if not prev_values.empty:
                    last_value = prev_values[name].iloc[-1]
                    # Create a new DataFrame with just the exact date
                    exact_df = pd.DataFrame({name: [last_value]}, index=exact_date_index)
                    all_dfs[i] = exact_df
    
    final_df = pd.concat(all_dfs, axis=1)
    final_df.index.name = 'date'
    final_df.reset_index(inplace=True)
    
    # Ensure we only have data within our date range
    mask = (final_df['date'] >= start_date) & (final_df['date'] <= end_date)
    final_df = final_df.loc[mask].copy()
    
    if not final_df.empty:
        print(f"\nFinal data summary:")
        print(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
        # Print data availability for each column
        for col in final_df.columns:
            if col != 'date':
                avail = (final_df[col].notna().sum() / len(final_df)) * 100
                print(f"{col}: {avail:.1f}% available data")
    else:
        print("Warning: No data was retrieved for the specified date range")
    
    return final_df


def validate_data(df: pd.DataFrame) -> None:
    """Validate the economic data for reasonable values."""
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Check for negative values where they shouldn't exist
    if (df['GDP'] < 0).any():
        print("Warning: Negative GDP values found")
    if (df['CPI'] < 0).any():
        print("Warning: Negative CPI values found")
    if (df['Unemployment_Rate'] < 0).any() or (df['Unemployment_Rate'] > 100).any():
        print("Warning: Unemployment rate outside reasonable range (0-100)")
    if (df['SP500'] < 0).any():
        print("Warning: Negative SP500 values found")


def insert_fred_data_manual(df: pd.DataFrame, conn_id: str) -> None:
    """Insert economic data into PostgreSQL database using Airflow connection."""
    if df.empty:
        raise ValueError("No data to insert")
    
    required_columns = ['date', 'GDP', 'Real_GDP', 'Unemployment_Rate', 'CPI', 'Fed_Funds_Rate', 'SP500']
    missing_columns = [col for col in required_columns if col.lower() not in [c.lower() for c in df.columns]]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    create_table_query = """
    CREATE TABLE IF NOT EXISTS us_economic_data_daily (
        date DATE PRIMARY KEY,
        gdp FLOAT,
        real_gdp FLOAT,
        unemployment_rate FLOAT,
        cpi FLOAT,
        fed_funds_rate FLOAT,
        sp500 FLOAT
    );
    """

    insert_query = """
    INSERT INTO us_economic_data_daily (
        date, gdp, real_gdp, unemployment_rate, cpi, fed_funds_rate, sp500
    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (date) DO NOTHING;
    """

    try:
        # Get connection using Airflow hook
        hook = PostgresHook(postgres_conn_id=conn_id)
        conn = hook.get_conn()
        cur = conn.cursor()
        
        # Create table
        cur.execute(create_table_query)

        # Prepare data for insertion
        records = [
            (
                row['date'],
                row.get('GDP'),
                row.get('Real_GDP'),
                row.get('Unemployment_Rate'),
                row.get('CPI'),
                row.get('Fed_Funds_Rate'),
                row.get('SP500')
            )
            for _, row in df.iterrows()
        ]

        # Insert data
        execute_batch(cur, insert_query, records)
        conn.commit()
        print(f"Successfully inserted {len(records)} economic metric records into the database.")
        
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


if __name__ == '__main__':
    try:
        print("Downloading and cleaning FRED data...")
        df = download_fred_data(start_date='2023-01-01', end_date='2023-03-01')
        print("\nSample of downloaded data:")
        print(df.head())
        print("\nData shape:", df.shape)

        validate_data(df)

        # Use Airflow connection ID
        insert_fred_data_manual(df, 'neon_db')
        print("\nAll operations completed successfully!")
    except Exception as e:
        print(f"\nError in main execution: {e}")