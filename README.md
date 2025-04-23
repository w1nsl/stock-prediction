# Stock Prediction Pipeline

A stock prediction system built with Apache Airflow that integrates multiple data sources including stock prices, economic indicators, and news sentiment to predict future stock movements.

## Overview

This project implements a complete ML pipeline that:

1. Extracts data from multiple sources:
   - Stock price data from Yahoo Finance
   - Economic indicators from FRED (Federal Reserve Economic Data)
   - News articles for sentiment analysis

2. Transforms the data:
   - Cleans and preprocesses stock price data
   - Performs sentiment analysis on news articles
   - Merges all data sources into a unified dataset

3. Loads the data into a PostgreSQL database

4. Implements a machine learning pipeline:
   - Feature engineering
   - Model training and evaluation
   - Prediction generation and storage

## Project Structure

```
.
├── dags/                       # Airflow DAG definitions
│   ├── project_DAG.py          # Main pipeline DAG
│   ├── article_sentiment.py    # News sentiment analysis module
│   ├── stock_price.py          # Stock price data processing
│   ├── us_economic_data.py     # Economic indicators module
│   ├── merged_data.py          # Data integration module
│   ├── ml_pipeline.py          # ML implementation
│   └── initialize_features.py  # Feature initialization
├── models/                     # Directory for saved ML models
├── notebooks/                  # Jupyter notebooks for development analysis
├── logs/                       # Airflow logs
├── plugins/                    # Airflow plugins
├── .env                        # Environment variables
├── Dockerfile                  # Docker image definition
├── docker-compose.yaml         # Docker Compose configuration
└── requirements.txt            # Python dependencies
```

## Features

- **Multi-Source Data Integration**: Combines stock prices, economic indicators, and news sentiment.
- **NLP-Based Sentiment Analysis**: Extracts sentiment from financial news articles.
- **Automated ML Pipeline**: Feature engineering, model training, and prediction in a single workflow.
- **Containerized Deployment**: Full Docker support for easy deployment.
- **Scalable Architecture**: Built on Airflow for scalable, distributed execution.
- **Modular Design**: Each component is isolated for easy maintenance and extension.

## Stocks Analyzed

The pipeline currently analyzes the following stocks:
- ADBE (Adobe)
- CMCSA (Comcast)
- QCOM (Qualcomm)
- GOOG (Google)
- PEP (PepsiCo)
- SBUX (Starbucks)
- COST (Costco)
- AMD (Advanced Micro Devices)
- INTC (Intel)
- PYPL (PayPal)

## Prerequisites
- Docker and Docker Compose
- Python 3.8+
- Apache Airflow

All required libraries are specified in the `requirements.txt` file.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-prediction.git
cd stock-prediction
```

2. Configure the environment variables:
```bash
# Edit .env file with your database credentials and API keys
cp .env.example .env
```
note: All relevant environment variables will be pre-filled and provided as part of the project submission to ensure reproducibility and ease of setup.

3. Build and start the Docker containers:
```bash
docker-compose up -d
```

4. Access the Airflow UI at http://localhost:8080 (username: airflow, password: airflow)

## Usage

1. The DAG is scheduled to run daily at midnight.
2. You can manually trigger a run from the Airflow UI.
3. Configure date ranges for analysis in the `project_DAG.py` file:
```python
# Specify date range. If not specified, will use DAG execution date 
START_DATE = "2023-12-16"  
END_DATE = "2023-12-16"
```
Due to limitations in the availability of recent news article data—specifically, the FNSPID dataset used in our pipeline has not been updated by its maintainer in a while. Thus we use the most recent entries available for demonstration purposes. As such, the START_DATE and END_DATE in project_DAG.py are currently set to "2023-12-16" to ensure that all pipeline components (news sentiment extraction, merging, modeling, and prediction) function end-to-end with complete data.

## Machine Learning Pipeline

The ML pipeline consists of the following steps:

1. **Feature Engineering**: Creates technical indicators, sentiment features, and economic features.
2. **Model Training**: Trains and evaluates models for each stock.
3. **Prediction Generation**: Makes predictions for future stock movements.
4. **Model Evaluation**: Tracks model performance over time.

Models and their performance metrics are stored in the database for comparison and analysis.

## Database Schema

The project uses a PostgreSQL database with the following main tables:

- `stock_prices`: Raw stock price data
- `economic_indicators`: Economic data from FRED
- `article_sentiment`: News sentiment analysis results
- `merged_stocks`: Integrated dataset with all features
- `ml_features`: Engineered features for ML
- `model_evaluations`: Model performance metrics
- `stock_predictions`: Prediction results

## Technology Stack

- **Apache Airflow**: Workflow orchestration
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning models
- **Transformers**: FinBERT models for sentiment analysis
- **yfinance**: Yahoo Finance data API
- **FRED API**: Economic data
- **Docker**: Containerization
- **PostgreSQL**: Data storage


## Stock Prediction Visualizations

This repository contains visualization tools for analyzing stock prediction data, including price movements, sentiment analysis, and economic indicators.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
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

## Acknowledgments

- Federal Reserve Economic Data (FRED)
- Yahoo Finance API
- HuggingFace Transformers 
