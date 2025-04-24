import streamlit as st
import sys
import os

# Basic page config
st.set_page_config(
    page_title="Matplotlib Test",
    page_icon="📊"
)

# Force matplotlib to use Agg backend
os.environ['MPLBACKEND'] = 'Agg'

st.title("Matplotlib Import Test")

# Information about Python
st.write(f"Python version: {sys.version}")
st.write(f"Python executable: {sys.executable}")
st.write(f"Python path: {sys.path}")

# Try to install matplotlib directly
try:
    st.write("Attempting to install matplotlib...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib==3.7.0"])
    st.success("Installation command completed")
except Exception as e:
    st.error(f"Installation error: {e}")

# Try to import matplotlib
try:
    st.write("Trying to import matplotlib...")
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    import matplotlib.pyplot as plt
    
    st.success(f"Successfully imported matplotlib {matplotlib.__version__}")
    
    # Try to create a simple plot
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
    st.pyplot(fig)
    
    st.success("Successfully created a matplotlib plot!")
except ImportError as e:
    st.error(f"Import error: {e}")
except Exception as e:
    st.error(f"Error: {e}") 