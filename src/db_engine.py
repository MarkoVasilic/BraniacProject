from sqlalchemy import create_engine
import pandas as pd

class DBEngine:
    def __init__(self, db_user='braniac', db_password='braniac', db_host='postgres', db_port='5432', db_name='braniac'):
        """
        Initialize the DBEngine.

        :param db_user: Database username.
        :param db_password: Database password.
        :param db_host: Database host.
        :param db_port: Database port.
        :param db_name: Database name.
        """
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.engine = self.create_engine()

    def create_engine(self):
        """
        Create and return a SQLAlchemy engine.

        :return: SQLAlchemy engine.
        """
        db_url = f'postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}'
        return create_engine(db_url)
    
    def save_dataframe(self, dataframe: pd.DataFrame, table_name: str, if_exists='replace', index=False):
        """
        Save a DataFrame to the database table.

        :param dataframe: Pandas DataFrame to be saved.
        :param table_name: Name of the database table.
        :param if_exists: How to behave if the table already exists.
        :param index: Whether to include the DataFrame index.
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
