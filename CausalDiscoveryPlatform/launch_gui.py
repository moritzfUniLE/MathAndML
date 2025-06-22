#!/usr/bin/env python3
"""
NOTEARS Web GUI Launcher
Simple script to launch the NOTEARS web interface with helpful instructions.
"""
import os
import sys
import webbrowser
import time
from threading import Timer

def check_requirements():
    """Check if required packages are installed."""
    try:
        import flask
        import flask_socketio
        import pandas
        import numpy
        import matplotlib
        import plotly
        import torch
        import networkx
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install requirements with: pip install -r requirements.txt")
        return False

def check_datasets():
    """Check if datasets directory exists."""
    if not os.path.exists('datasets'):
        print("Warning: 'datasets' directory not found.")
        print("Please run 'python create_test_datasets.py' first to generate test datasets.")
        return False
    
    datasets = [d for d in os.listdir('datasets') if os.path.isdir(os.path.join('datasets', d))]
    print(f"Found {len(datasets)} datasets: {', '.join(datasets)}")
    return True

def open_browser():
    """Open the web browser after a short delay."""
    time.sleep(2)  # Wait for server to start
    webbrowser.open('http://localhost:5000')

def main():
    """Main launcher function."""
    print("NOTEARS Web GUI Launcher")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    print("✓ All requirements satisfied")
    
    # Check datasets
    check_datasets()
    
    print("\nLaunching NOTEARS Web GUI...")
    print("- Server will start on http://localhost:5000")
    print("- Your browser will open automatically")
    print("- Press Ctrl+C to stop the server")
    print()
    
    # Schedule browser opening
    Timer(2.0, open_browser).start()
    
    # Launch the web application
    try:
        from notears_web_gui import NOTEARSWebApp
        app = NOTEARSWebApp()
        app.socketio.run(app.app, host='localhost', port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nShutting down NOTEARS Web GUI...")
    except Exception as e:
        print(f"Error starting web GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()