from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import os

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save the model (optional)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, 'logistic_regression_model.joblib')

    # Save the model using joblib
    from joblib import dump
    dump(model, model_file)

if __name__ == '__main__':
    from data_preprocessing import preprocess_data
    from data_exploration import load_data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_logistic_regression(X_train, y_train)
    evaluate_model(model, X_test, y_test)
