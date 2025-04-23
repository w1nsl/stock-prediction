import webbrowser
import urllib.parse

# Define the parameters you want to prefill
params = {
    "stock_symbols": "GOOG,AMD,COST,PYPL,QCOM,ADBE,PEP,CMCSA,INTC,SBUX",
    "selected_stock": "GOOG",
    "start_date": "2023-06-16",
    "end_date": "2023-12-31"
}

# Build the URL with query parameters
base_url = "http://localhost:8502/"
query_string = urllib.parse.urlencode(params)
url = f"{base_url}?{query_string}"

print(f"Opening dashboard with the following URL:")
print(url)

# Open the URL in the default browser
webbrowser.open(url)

print("\nNote: Make sure your Streamlit dashboard is already running on port 8502.")
print("If not, start it first with: python run_dashboards.py --dashboard prediction") 