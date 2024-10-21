import pandas as pd
import plotly.express as px
import os

def plot_time_vs_amount(df):
    fig = px.scatter(df, x='Time', y='Amount', color='Class', title='Transaction Amount over Time')

    # Create results directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save plot
    output_file = os.path.join(results_dir, 'time_vs_amount.html')
    fig.write_html(output_file)

if __name__ == '__main__':
    from data_exploration import load_data
    df = load_data()
    plot_time_vs_amount(df)
