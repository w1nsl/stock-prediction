import streamlit as st
import os
import sys

# Add the current directory to the path so imports work correctly
sys.path.append(os.path.dirname(__file__))

# Redirect to the actual application
import visualizations.prediction_dashboard 