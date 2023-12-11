from sqlalchemy import create_engine
import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker

class DBEngine:
    def __init__(self):
        """
        Initialize the DBEngine.

        Attributes:
            db_user (str): Database username.
            db_password (str): Database password.
            db_host (str): Database host.
            db_port (str): Database port.
            db_name (str): Database name.
            engine: SQLAlchemy engine.
            session: SQLAlchemy session.
        """
        load_dotenv()
        self.db_user: str = os.getenv('DB_USER', 'braniac')
        self.db_password: str = os.getenv('DB_PASSWORD', 'braniac')
        self.db_host: str = os.getenv('DB_HOST', 'postgres')
        self.db_port: str = os.getenv('DB_PORT', '5432')
        self.db_name: str = os.getenv('DB_NAME', 'braniac')
        self.engine = self.create_engine()
        self.session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def __new__(cls):
        """
        Implement the Singleton pattern for DBEngine.

        Returns:
            DBEngine: Singleton instance of DBEngine.
        """
        if not hasattr(cls, 'instance'):
            cls.instance = super(DBEngine, cls).__new__(cls)
        return cls.instance

    def create_engine(self) -> create_engine:
        """
        Create and return a SQLAlchemy engine.

        Returns:
            create_engine: SQLAlchemy engine.
        """
        db_url = f'postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}'
        return create_engine(db_url)
    
    def get_db(self):
        """
        Create a database session.

        Yields:
            Session: SQLAlchemy database session.
        """
        db = self.session()
        try:
            yield db
        finally:
            db.close()
    
    def save_dataframe(self, dataframe: pd.DataFrame, table_name: str, if_exists='replace', index=False) -> None:
        """
        Save a DataFrame to the database table.

        Args:
            dataframe (pd.DataFrame): Pandas DataFrame to be saved.
            table_name (str): Name of the database table.
            if_exists (str): How to behave if the table already exists.
            index (bool): Whether to include the DataFrame index.
        """
        dataframe.to_sql(table_name, con=self.engine, if_exists=if_exists, index=index)

    def fetch_data(self, table_name: str) -> pd.DataFrame:
        """
        Fetch data from a PostgreSQL table.

        Args:
            table_name (str): Name of the table to fetch data from.

        Returns:
            pd.DataFrame: DataFrame containing the fetched data.
        """
        try:
            query = f'SELECT * FROM {table_name}'
            df = pd.read_sql_query(query, self.engine)
            return df
        except Exception as e:
            print(f"Error fetching data from {table_name}: {e}")
            return pd.DataFrame()
