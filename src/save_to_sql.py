import pandas as pd
from sqlalchemy import create_engine
import os

def save_to_sql():
    # Define file paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    database_dir = os.path.join(base_dir, 'database')

    # Ensure the database directory exists
    os.makedirs(database_dir, exist_ok=True)

    # File paths
    csv_file = os.path.join(data_dir, 'creditcard.csv')
    db_file = os.path.join(database_dir, 'creditcard.db')

    # Load CSV into a DataFrame
    df = pd.read_csv(csv_file)

    # Create an SQLite database and engine using SQLAlchemy
    engine = create_engine(f'sqlite:///{db_file}')
    df.to_sql('transactions', engine, index=False, if_exists='replace')

if __name__ == '__main__':
    save_to_sql()
