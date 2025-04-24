import os
import sys
import subprocess
import signal
import platform
import time
import streamlit as st

# Add the current directory to the path so we can import from subdirectories
sys.path.insert(0, os.path.dirname(__file__))

# Check if we're in a cloud deployment
def is_cloud_deployment():
    # Look for environment variables commonly set in cloud environments
    return any([
        os.environ.get('DYNO') is not None,  # Heroku
        os.environ.get('STREAMLIT_SHARING') is not None,  # Streamlit sharing
        os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None,  # AWS
        os.environ.get('WEBSITE_SITE_NAME') is not None,  # Azure
        'render' in os.environ.get('RENDER_SERVICE', '').lower(),  # Render
    ])

# Check if running in a cloud environment
is_cloud = is_cloud_deployment()

if is_cloud:
    # In cloud environments, import the dashboard directly
    # This avoids subprocess issues in some cloud environments
    try:
        # Set a timeout for database connections
        os.environ['DB_CONNECT_TIMEOUT'] = '10'  # 10 seconds
        os.environ['DATABASE_MAX_RETRIES'] = '2'  # Only try 2 times
        
        # Import the dashboard module (it will run automatically)
        st.spinner("Starting dashboard (cloud mode)...")
        from visualizations.prediction_dashboard import *
    except Exception as e:
        st.error(f"Error initializing dashboard: {str(e)}")
        st.info("If you're experiencing database connection issues, please check your database credentials and network connectivity.")
        
        # Show detailed error for debugging
        import traceback
        st.code(traceback.format_exc())
else:
    # For local environments, use the subprocess approach
    def run_prediction_dashboard():
        """Run the prediction dashboard as a separate process"""
        script_path = os.path.join(os.path.dirname(__file__), "visualizations", "prediction_dashboard.py")
        
        # Add additional environment variables
        env = os.environ.copy()
        env['DB_CONNECT_TIMEOUT'] = '10'  # 10 seconds
        
        # Build command
        cmd = ["streamlit", "run", script_path]
        
        # Run the command
        process = subprocess.Popen(cmd, env=env)
        
        try:
            # Wait for the process to complete
            process.wait()
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            process.send_signal(signal.SIGINT)
            process.wait()

    if __name__ == "__main__":
        run_prediction_dashboard() 