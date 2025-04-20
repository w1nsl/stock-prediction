# Stock Prediction Visualizations

This repository contains visualization tools for analyzing stock prediction data, including price movements, sentiment analysis, and economic indicators.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure your database connection by creating a `.env` file with the following variables:
   ```
   DB_HOST=your_host
   DB_NAME=your_database
   DB_USER=your_username
   DB_PASSWORD=your_password
   DB_PORT=5432
   DB_SSLMODE=require
   ```

## Project Structure

```
.
├── dags/                 # Airflow DAGs for data processing
├── visualizations/       # Visualization package
│   ├── __init__.py       # Package initialization
│   ├── core.py           # Core visualization functions
│   ├── predictions.py    # Prediction analysis functions
│   ├── dashboard.py      # Main Streamlit dashboard
│   └── prediction_dashboard.py  # Prediction analysis dashboard
├── data/                 # Data files (created automatically)
├── run_dashboards.py     # Script to run dashboards
└── requirements.txt      # Project dependencies
```

## Components

### 1. Visualization Package (`visualizations/`)

The `visualizations` package provides functions for creating various types of visualizations:

#### Core Visualizations (`core.py`)

- Stock Price Candlestick Chart with Volume
- Sentiment Analysis Visualization
- Economic Indicators Correlation Heatmap
- Combined Economic Indicators Dashboard
- Prediction Performance Visualization
- Feature Importance Visualization

**Usage:**
```python
from visualizations import load_data_from_db, plot_stock_candlestick

# Load data
df = load_data_from_db(stock_symbol='AAPL', start_date='2023-01-01')

# Create candlestick chart
fig = plot_stock_candlestick(df, 'AAPL')
fig.write_html("visualizations/static/AAPL_candlestick.html")
```

#### Prediction Analysis (`predictions.py`)

This module focuses on analyzing model predictions:

- Actual vs. Predicted Price Comparison
- Prediction Error Analysis
- Error Distribution
- Accuracy vs. Prediction Horizon
- Performance by Market Volatility

**Usage:**
```python
from visualizations import load_predictions, plot_prediction_comparison

# Load prediction data
df = load_predictions('AAPL', start_date='2023-01-01')

# Create comparison chart
fig = plot_prediction_comparison(df, 'AAPL')
fig.write_html("visualizations/static/AAPL_prediction_comparison.html")
```

### 2. Interactive Dashboards

#### Main Dashboard (`dashboard.py`)

An interactive Streamlit dashboard that combines stock data visualizations with sentiment and economic indicators.

**Features:**
- Stock selection
- Date range filtering
- Interactive charts and metrics
- Multiple visualization tabs

#### Prediction Dashboard (`prediction_dashboard.py`)

An interactive Streamlit dashboard specifically for analyzing prediction performance.

**Features:**
- Actual vs predicted price comparison
- Error analysis and distribution
- Performance metrics (RMSE, MAE, MAPE, R², Direction Accuracy)
- Advanced analysis (prediction horizon, volatility)

### 3. Running the Dashboards

You can run the dashboards easily using the provided script:

```
# Run the main dashboard (default)
python run_dashboards.py

# Run the prediction dashboard
python run_dashboards.py --dashboard prediction

# Run both dashboards simultaneously
python run_dashboards.py --dashboard all
```

Alternatively, you can run them directly with Streamlit:

```
# Main dashboard
streamlit run visualizations/dashboard.py

# Prediction dashboard
streamlit run visualizations/prediction_dashboard.py
```

## Example Visualizations

The repository includes functions for generating:

1. **Stock Price Analysis**
   - Candlestick charts with volume
   - Moving averages
   - Price statistics

2. **Sentiment Analysis**
   - Sentiment score over time
   - Correlation with stock price
   - Positive/negative ratio distribution
   - Article count trends

3. **Economic Indicators**
   - GDP, Unemployment, CPI, Fed Funds Rate
   - S&P 500 correlation
   - Multi-indicator comparison
   - Lagged impact analysis

4. **Prediction Performance**
   - Actual vs. predicted comparison
   - Error metrics (RMSE, MAE, MAPE)
   - Directional accuracy
   - Error distribution visualization

## Output Files

Generated visualizations are saved to the `visualizations/static/` directory in HTML format for interactive viewing. 