import argparse
import os
import sys
import subprocess
import webbrowser
import time
import signal
import threading

def run_dashboard(dashboard_name, open_browser=True):
    """Run the specified dashboard"""
    if dashboard_name == "main":
        script = "visualizations/dashboard.py"
        port = 8501
    elif dashboard_name == "prediction":
        script = "visualizations/prediction_dashboard.py"
        port = 8502
    else:
        print(f"Unknown dashboard: {dashboard_name}")
        return
    
    # Build command - add --server.headless=true to prevent Streamlit auto-browser
    cmd = ["streamlit", "run", script, "--server.port", str(port), "--server.headless", "true"]
    
    # Run the command
    process = subprocess.Popen(cmd)
    
    # Open the browser after a short delay if requested
    if open_browser:
        threading.Timer(2, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    
    try:
        # Wait for the process to complete
        process.wait()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        process.send_signal(signal.SIGINT)
        process.wait()

def run_all_dashboards(open_browser=True):
    """Run all dashboards"""
    # Build commands with --server.headless=true to prevent auto-browser
    main_cmd = ["streamlit", "run", "visualizations/dashboard.py", "--server.port", "8501", "--server.headless", "true"]
    pred_cmd = ["streamlit", "run", "visualizations/prediction_dashboard.py", "--server.port", "8502", "--server.headless", "true"]
    
    # Run the commands
    main_process = subprocess.Popen(main_cmd)
    pred_process = subprocess.Popen(pred_cmd)
    
    # Open browsers after a short delay if requested
    if open_browser:
        threading.Timer(2, lambda: webbrowser.open("http://localhost:8501")).start()
        threading.Timer(3, lambda: webbrowser.open("http://localhost:8502")).start()
    
    try:
        # Wait for processes to complete
        main_process.wait()
        pred_process.wait()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        main_process.send_signal(signal.SIGINT)
        pred_process.send_signal(signal.SIGINT)
        main_process.wait()
        pred_process.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stock prediction dashboards")
    parser.add_argument("--dashboard", choices=["main", "prediction", "all"], default="main",
                        help="Which dashboard to run (default: main)")
    parser.add_argument("--no-browser", action="store_true", 
                        help="Don't automatically open a browser window")
    
    args = parser.parse_args()
    
    # Determine if browser should be opened
    open_browser = not args.no_browser
    
    if args.dashboard == "all":
        run_all_dashboards(open_browser=open_browser)
    else:
        run_dashboard(args.dashboard, open_browser=open_browser) 