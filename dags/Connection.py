from airflow.providers.postgres.hooks.postgres import PostgresHook
from psycopg2.extensions import cursor

CONN_ID = 'postgres_default'

class Connection:
    def __init__(self) -> None:
        self.conn = None
        self.cur = None

    def connect(self, conn_id: str = CONN_ID) -> None:
        hook = PostgresHook(postgres_conn_id=conn_id)
        self.conn = hook.get_conn()
        self.cur = self.conn.cursor()

    def exit(self) -> bool:
        self.cur.close()
        self.conn.close()
    
    def initialise_db(self, connection=None, conn_id: str = CONN_ID) -> str:
        """Initialize database tables
        Args:
            connection: Optional Connection instance (not used, kept for compatibility)
            conn_id: Airflow connection ID for PostgreSQL
        Returns:
            conn_id: The connection ID used
        """
        self.connect(conn_id=conn_id)
        query = '''
            CREATE table if not EXISTS Company ( 
                company_id SERIAL PRIMARY KEY, 
                company_name VARCHAR(255) NOT NULL, 
                ticker_symbol VARCHAR(10) UNIQUE NOT NULL, 
                sector VARCHAR(100) 
            ); 
            
            CREATE TABLE if not exists Price ( 
                price_id SERIAL PRIMARY KEY, 
                company_id INT REFERENCES Company(company_id), 
                date TIMESTAMP NOT NULL, 
                open_price DECIMAL(12, 2), 
                close_price DECIMAL(12, 2), 
                high_price DECIMAL(12, 2), 
                low_price DECIMAL(12, 2), 
                dividends DECIMAL(12, 2), 
                volume INT, 
                stock_splits INT 
            ); 
            
            CREATE TABLE if not exists Tweet_Headline ( 
                tweet_id SERIAL PRIMARY KEY, 
                company_id INT REFERENCES Company(company_id), 
                content TEXT, 
                date TIMESTAMP NOT NULL 
            ); 
            
            CREATE TABLE if not exists Analysis (
                analysis_id SERIAL PRIMARY KEY,
                tweet_id INT REFERENCES Tweet_Headline(tweet_id),
                sentiment_score FLOAT,
                sentiment_label TEXT
            );
        '''
        self.cur.execute(query)
        self.conn.commit()
        self.exit()
        return conn_id