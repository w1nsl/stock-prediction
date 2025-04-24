"""
Main Streamlit entry point with proper configuration for Streamlit Cloud
"""
import streamlit as st
import os
import sys

# Configure page
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set matplotlib backend before any other imports - use Agg for headless environments
os.environ['MPLBACKEND'] = 'Agg'

try:
    # Force matplotlib configuration
    import matplotlib
    matplotlib.use('Agg')  # Use Agg backend which is compatible with Streamlit Cloud (no GUI required)
    
    # Explicitly import pyplot to ensure it gets configured properly
    import matplotlib.pyplot as plt
    
    # Set non-interactive mode explicitly
    plt.ioff()
    
    # Configure matplotlib for streamlit
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 100
    
    # Check that the backend is configured correctly
    st.sidebar.write(f"Using matplotlib backend: {matplotlib.get_backend()}")
    
    # Now add current directory to path
    sys.path.insert(0, os.path.dirname(__file__))
    
    # Import the actual dashboard with the correct environment set up
    from visualizations import prediction_dashboard
    
except ImportError as e:
    st.error(f"Failed to import required libraries: {e}")
    
    # Provide more detailed error information
    import traceback
    st.code(traceback.format_exc())
    
    # Show a more user-friendly message
    st.warning("""
    ## Dashboard Import Error
    
    The dashboard failed to load due to missing dependencies. Here are some possible solutions:
    
    1. If you're seeing an error about matplotlib or plotly, try using the streamlit_viz.py file instead
    2. If you're deploying on Streamlit Cloud, there may be permission issues with installing certain packages
    
    For a simplified version that works on Streamlit Cloud, try deploying streamlit_viz.py instead.
    """)
    
    # Try to show system information
    st.subheader("System Information")
    st.write(f"Python version: {sys.version}")
    st.write(f"Python executable: {sys.executable}")
    st.write(f"Python path: {sys.path}") 