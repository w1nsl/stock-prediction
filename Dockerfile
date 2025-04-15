FROM apache/airflow:2.10.5-python3.11

# Set environment variables
ENV AIRFLOW_HOME=/opt/airflow

# Install OS-level dependencies if needed (e.g., for yfinance or NLP tools)
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy custom DAG files and modules into the container
COPY ./dags /opt/airflow/dags
COPY ./plugins /opt/airflow/plugins

# Install Python packages
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt
