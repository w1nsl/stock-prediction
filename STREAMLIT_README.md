# Stock Prediction Dashboard - Streamlit Deployment

## Deployment Instructions

### 1. Set up on Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Create a new app
3. Connect to your GitHub repository
4. Set the main file path to `streamlit_app.py`
5. Deploy

### 2. Configure Secrets in Streamlit Cloud

After deploying, you need to configure your database secrets:

1. Go to your app settings
2. Navigate to the "Secrets" section
3. Add the following configuration:

```toml
[db]
host = "your-database-host"
name = "your-database-name"
user = "your-database-user"
password = "your-database-password"
port = 5432
sslmode = "require"
```

Replace the placeholder values with your actual database credentials.

### 3. Local Development

To run the dashboard locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

## Files for Deployment

- `streamlit_app.py`: Main entry point for Streamlit Cloud
- `requirements.txt`: Python dependencies
- `.streamlit/config.toml`: Streamlit configuration 
- `.streamlit/secrets.toml`: Database credentials (local development only, not committed to Git)
- `visualizations/prediction_dashboard.py`: Main dashboard code

## Database Connection

The app is configured to:
1. Use Streamlit secrets in cloud deployment
2. Fall back to environment variables for local development
3. Generate sample data if database connection fails

## Troubleshooting

If you encounter database connection issues:

1. Verify your database credentials in Streamlit Cloud secrets
2. Check that your database accepts connections from Streamlit Cloud IP addresses
3. Ensure your database is active and not in hibernation mode
4. The app will generate sample data if database connection fails 