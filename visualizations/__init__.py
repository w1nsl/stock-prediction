"""
Stock Prediction Visualization Package

This package provides tools for visualizing stock data, sentiment analysis,
economic indicators, and model predictions.
"""

from visualizations.core import (
    get_db_connection,
    load_data_from_db,
    plot_stock_candlestick,
    plot_sentiment_analysis,
    plot_correlation_heatmap,
    plot_economic_dashboard,
    plot_prediction_performance,
    plot_feature_importance
)

from visualizations.predictions import (
    load_predictions,
    plot_prediction_comparison,
    plot_error_analysis,
    plot_error_distribution,
    calculate_metrics,
    plot_accuracy_vs_horizon,
    plot_performance_by_volatility,
    sample_predictions_to_csv,
    load_feature_importance
)

__all__ = [
    # Database functions
    'get_db_connection',
    'load_data_from_db',
    'load_predictions',
    'load_feature_importance',
    
    # Core visualizations
    'plot_stock_candlestick',
    'plot_sentiment_analysis',
    'plot_correlation_heatmap',
    'plot_economic_dashboard',
    'plot_prediction_performance',
    'plot_feature_importance',
    
    # Prediction analysis
    'plot_prediction_comparison',
    'plot_error_analysis',
    'plot_error_distribution',
    'calculate_metrics',
    'plot_accuracy_vs_horizon',
    'plot_performance_by_volatility',
    'sample_predictions_to_csv'
] 