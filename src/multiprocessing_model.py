# src/multiprocessing_model.py

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool
import numpy as np
import os

def evaluate_model(params):
    X_train, y_train, cv = params
    model = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=1)  # Use single thread for each model training
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=1)  # Single-threaded cross-validation
    return np.mean(scores)

if __name__ == '__main__':
    from data_preprocessing import preprocess_data
    from data_exploration import load_data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Parameters for multiprocessing
    params = (X_train, y_train, 5)  # 5-fold cross-validation

    with Pool() as pool:
        result = pool.map(evaluate_model, [params])

    print(f"Average F1 Score from Cross-Validation: {result[0]}")
