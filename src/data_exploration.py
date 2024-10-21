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

    # Amount distribution plot
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Amount'], bins=50, kde=True)
    plt.title('Amount Distribution')
    plt.xlabel('Amount')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(results_dir, 'amount_distribution.png'))
    plt.close()

    # Time distribution plot
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Time'], bins=50, kde=True)
    plt.title('Time Distribution')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(results_dir, 'time_distribution.png'))
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(results_dir, 'correlation_heatmap.png'))
    plt.close()

    # Amount by Class boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Class', y='Amount', data=df)
    plt.title('Amount by Class Boxplot')
    plt.savefig(os.path.join(results_dir, 'amount_by_class_boxplot.png'))
    plt.close()

if __name__ == '__main__':
    df = load_data()
    explore_data(df)
