import pandas as pd
from sqlalchemy import create_engine
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    # Define file paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    database_dir = os.path.join(base_dir, 'database')

    # File paths
    db_file = os.path.join(database_dir, 'creditcard.db')

    # Create an SQLite database engine
    engine = create_engine(f'sqlite:///{db_file}')

    # Load data into DataFrame
    df = pd.read_sql('SELECT * FROM transactions', engine)
    return df

def explore_data(df):
    # Print basic information
    print(df.head())
    print(df.describe())
    print(df.info())

    # Check for missing values
    print(df.isnull().sum())

    # Create results directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Class distribution plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution')
    plt.savefig(os.path.join(results_dir, 'class_distribution.png'))
    plt.close()

if __name__ == '__main__':
    df = load_data()
    explore_data(df)
