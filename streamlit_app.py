import os
import sys
import subprocess
import signal

# Add the current directory to the path so we can import from subdirectories
sys.path.insert(0, os.path.dirname(__file__))

def run_prediction_dashboard():
    """Run the prediction dashboard as a separate process"""
    script_path = os.path.join(os.path.dirname(__file__), "visualizations", "prediction_dashboard.py")
    
    # Build command
    cmd = ["streamlit", "run", script_path]
    
    # Run the command
    process = subprocess.Popen(cmd)
    
    try:
        # Wait for the process to complete
        process.wait()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        process.send_signal(signal.SIGINT)
        process.wait()

if __name__ == "__main__":
    run_prediction_dashboard() 