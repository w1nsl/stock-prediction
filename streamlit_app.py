import streamlit as st
import os
import sys

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
        st.write("Please check the requirements.txt file and ensure all dependencies are installed.")
        st.code(f"Technical details: {str(e)}")
        
        # Show installed packages for debugging
        import pkg_resources
        installed_packages = [f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set]
        with st.expander("Installed packages"):
            st.code("\n".join(sorted(installed_packages))) 