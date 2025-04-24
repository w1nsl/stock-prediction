import streamlit as st
import os
import sys
import subprocess

# Basic page configuration
st.set_page_config(
    page_title="Stock Prediction Dashboard (Minimal)",
    page_icon="📈",
    layout="wide"
)

st.title("Stock Prediction Dashboard")
st.write("This is a minimal version of the dashboard for testing deployment.")

# Attempt to import common libraries and report success/failure
libraries = [
    "pandas", 
    "numpy", 
    "matplotlib", 
    "plotly", 
    "seaborn", 
    "sklearn", 
    "psycopg2"
]

results = {}

for lib in libraries:
    try:
        exec(f"import {lib}")
        results[lib] = "✅ Successfully imported"
        
        # For matplotlib, try to create a simple plot
        if lib == "matplotlib":
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
                st.pyplot(fig)
                results[lib] += " and created a test plot"
            except Exception as plot_err:
                results[lib] += f" but error creating plot: {plot_err}"
                
        # For plotly, try to create a simple plot
        if lib == "plotly":
            try:
                import plotly.express as px
                df = px.data.iris()
                fig = px.scatter(df, x="sepal_width", y="sepal_length")
                st.plotly_chart(fig)
                results[lib] += " and created a test plot"
            except Exception as plot_err:
                results[lib] += f" but error creating plot: {plot_err}"
                
    except ImportError as e:
        results[lib] = f"❌ Failed to import: {e}"
    except Exception as e:
        results[lib] = f"❌ Error: {e}"

# Display the results
st.subheader("Library Import Test Results")
for lib, result in results.items():
    st.write(f"**{lib}**: {result}")

# Show system information
st.subheader("System Information")
st.write(f"Python version: {sys.version}")

# Show installed packages
try:
    result = subprocess.check_output([sys.executable, "-m", "pip", "list"]).decode('utf-8')
    with st.expander("Installed packages"):
        st.code(result)
except Exception as pip_error:
    st.write(f"Could not list installed packages: {pip_error}") 