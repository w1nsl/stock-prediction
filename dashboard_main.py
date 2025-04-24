"""
Main entry point for the dashboard with proper matplotlib configuration
"""
import streamlit as st
import os
import sys
from threading import RLock

# Configure page
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set matplotlib backend before any other imports
os.environ['MPLBACKEND'] = 'Agg'

# Create a global lock for matplotlib operations
matplotlib_lock = RLock()

try:
    # Configure matplotlib properly
    import matplotlib
    matplotlib.use('Agg')  # Use Agg backend which is compatible with Streamlit Cloud
    import matplotlib.pyplot as plt
    
    # Create a wrapper for st.pyplot that uses the lock
    def pyplot_with_lock(fig=None, **kwargs):
        with matplotlib_lock:
            return st.pyplot(fig, **kwargs)
    
    # Replace the standard st.pyplot with our locked version
    st._pyplot = st.pyplot
    st.pyplot = pyplot_with_lock
    
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