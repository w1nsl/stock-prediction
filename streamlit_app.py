import os
import sys

# Add the current directory to the path so we can import from subdirectories
sys.path.insert(0, os.path.dirname(__file__))

# Import the prediction dashboard module
from visualizations.prediction_dashboard import *

# The code from prediction_dashboard.py will automatically execute
# so we don't need to add any additional code here 