from airflow.decorators import dag, task
from datetime import datetime
from Connection import Connection  # Make sure Connection.py is in the same folder or importable path

CONN_ID = 'is3107_db'  # Replace with your actual Airflow Postgres connection ID if different

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0
}

@dag(
    dag_id='stock_price_prediction',
    default_args=default_args,
    schedule_interval=None,  # Manual trigger only for now
    catchup=False,
    tags=['initialisation']
)
def init_postgres_database():
    
    @task
    def setup_database():
        conn = Connection()
        conn.connect(conn_id=CONN_ID)
        conn.init_db()
        return "Database initialized successfully"
    
    setup_database()

projectDAG = init_postgres_database()
