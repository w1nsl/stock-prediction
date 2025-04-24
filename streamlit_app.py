import streamlit as st
import os
import sys
import subprocess

# Page configuration
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the current directory to the path so imports work correctly
sys.path.insert(0, os.path.dirname(__file__))

# Display loading message while the app initializes
with st.spinner("Loading dashboard... This may take a few moments."):
    try:
        # Import the actual dashboard
        import visualizations.prediction_dashboard
    except ImportError as e:
        st.error(f"Error loading the dashboard: {e}")
        st.write("If you're seeing this error, there might be a missing dependency.")
        
        # Show Python version for debugging
        st.write(f"Python version: {sys.version}")
        
        # Try to list installed packages using pip
        try:
            result = subprocess.check_output([sys.executable, "-m", "pip", "list"]).decode('utf-8')
            with st.expander("Installed packages"):
                st.code(result)
        except Exception as pip_error:
            st.write(f"Could not list installed packages: {pip_error}")
            
        # Show environment variables (excluding sensitive ones)
        env_vars = {k: v for k, v in os.environ.items() 
                    if not any(sensitive in k.lower() for sensitive in 
                              ['password', 'secret', 'token', 'key'])}
        with st.expander("Environment Variables"):
            st.json(env_vars) 