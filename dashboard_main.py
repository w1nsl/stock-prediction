"""
Main entry point for the dashboard with proper matplotlib configuration
"""
import streamlit as st
import os
import sys

# Configure page
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set matplotlib backend before any other imports
os.environ['MPLBACKEND'] = 'Agg'

try:
    # Configure matplotlib properly
    import matplotlib
    matplotlib.use('Agg')  # Use Agg backend which is compatible with Streamlit Cloud
    import matplotlib.pyplot as plt
    
    # Set non-interactive mode explicitly
    plt.ioff()
    
    # Configure matplotlib for streamlit
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 100
    
    # Now add current directory to path
    sys.path.insert(0, os.path.dirname(__file__))
    
    # Import the dashboard 
    from visualizations import dashboard
    
except ImportError as e:
    st.error(f"Failed to import required libraries: {e}")
    import traceback
    st.code(traceback.format_exc())
    
    # Display system information for debugging
    st.subheader("System Information")
    st.write(f"Python version: {sys.version}")
    st.write(f"Python executable: {sys.executable}")
    st.write(f"Python path: {sys.path}") 