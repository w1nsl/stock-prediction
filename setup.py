from setuptools import setup, find_packages

setup(
    name="stock-prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "plotly",
        "scikit-learn",
        "psycopg2-binary",
        "python-dotenv",
    ],
) 