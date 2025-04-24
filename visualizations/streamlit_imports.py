"""
Import and configure libraries for Streamlit compatibility
"""
import os
import streamlit as st

# Configure matplotlib for Streamlit
def configure_matplotlib():
    """Set matplotlib config for Streamlit compatibility"""
    try:
        # Set the backend to 'Agg' which is non-interactive and should work on Streamlit Cloud
        os.environ['MPLBACKEND'] = 'Agg'
        
        # Import matplotlib after setting the backend
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Configure matplotlib parameters for better integration
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 100
        
        return True
    except Exception as e:
        st.error(f"Error configuring matplotlib: {e}")
        return False

# Function to display matplotlib figures in Streamlit
def show_pyplot(fig):
    """
    Correctly display a matplotlib figure in Streamlit
    
    Args:
        fig: Matplotlib figure object
    """
    try:
        # Use st.pyplot which is the proper way to display matplotlib figures
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error displaying matplotlib figure: {e}")
        st.write("If you're seeing this error, Streamlit might be having issues with matplotlib.")
        
# Configure matplotlib when this module is imported
configure_matplotlib() 