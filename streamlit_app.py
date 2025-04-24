import os
import sys
import time
import threading
import streamlit as st

# Add the current directory to the path so we can import from subdirectories
sys.path.insert(0, os.path.dirname(__file__))

# Global variable to track initialization
initialized = False

def is_cloud_deployment():
    # Look for environment variables commonly set in cloud environments
    return any([
        os.environ.get('DYNO') is not None,  # Heroku
        os.environ.get('STREAMLIT_SHARING') is not None,  # Streamlit sharing
        os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None,  # AWS
        os.environ.get('WEBSITE_SITE_NAME') is not None,  # Azure
        'render' in os.environ.get('RENDER_SERVICE', '').lower(),  # Render
    ])

def initialize_app_in_background():
    """Initialize the app in a background thread to not block health checks"""
    global initialized
    try:
        # Set timeouts for database connections
        os.environ['DB_CONNECT_TIMEOUT'] = '5'  # 5 seconds
        os.environ['DATABASE_MAX_RETRIES'] = '1'  # Only try once
        
        # Import here to delay loading until after health check passes
        from visualizations.predictions import load_predictions
        
        # Mark as initialized
        initialized = True
    except Exception as e:
        st.error(f"Error initializing app: {str(e)}")
        import traceback
        with open("error_log.txt", "w") as f:
            f.write(traceback.format_exc())
        initialized = "error"

# Start initialization in background
if not initialized:
    init_thread = threading.Thread(target=initialize_app_in_background)
    init_thread.daemon = True
    init_thread.start()

# Main app
def main():
    # This must be the first Streamlit command
    st.set_page_config(
        page_title="Stock Price Prediction Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    # Header
    st.title("Stock Price Prediction Dashboard")
    
    # Show a spinner while initializing
    if initialized == False:
        st.info("Dashboard is initializing, please wait...")
        st.spinner("Loading dependencies and connecting to database...")
        # Add a placeholder that will be replaced when initialization is complete
        placeholder = st.empty()
        for i in range(10):  # Try for about 10 seconds
            if initialized:
                break
            time.sleep(1)
            placeholder.text(f"Still initializing... ({i+1}s)")
        
        if not initialized or initialized == "error":
            st.warning("Dashboard initialization is taking longer than expected.")
            st.info("You can continue waiting or refresh the page.")
            st.write("If problems persist, check the database connection.")
            return
    
    # If we're here, initialization is complete - import and run the dashboard
    try:
        # Set environment variable to tell dashboard not to call set_page_config
        os.environ['SKIP_PAGE_CONFIG'] = 'true'
        
        # Import dashboard module selectively to avoid the st.set_page_config conflict
        # Import specific functions we need instead of entire module
        from visualizations.prediction_dashboard import load_cached_predictions
        
        # Run the dashboard logic manually
        import visualizations.prediction_dashboard
        # Remove the set_page_config attribute to prevent it from being called
        visualizations.prediction_dashboard.st.set_page_config = lambda **kwargs: None
        
        # Now import the rest of the module which will run without calling set_page_config
        import importlib
        importlib.reload(visualizations.prediction_dashboard)
    except Exception as e:
        st.error(f"Error running dashboard: {str(e)}")
        st.info("Try refreshing the page. If the problem persists, please contact support.")
        
        # Show detailed error only in development environment
        if not is_cloud_deployment():
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main() 