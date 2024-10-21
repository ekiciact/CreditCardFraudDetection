# Credit Card Fraud Detection

This project aims to develop a machine learning model to predict fraudulent credit card transactions. The dataset used is highly imbalanced, with a very small percentage of fraudulent transactions, making it an excellent challenge for data scientists to apply various methods to improve model accuracy.

## Project Structure

The project is structured as follows:

```
credit-card-fraud-detection/
├── data/
│   └── creditcard.csv
├── database/
│   └── creditcard.db
├── src/
│   ├── save_to_sql.py
│   ├── data_exploration.py
│   ├── data_visualization.py
│   ├── data_preprocessing.py
│   ├── model_development.py
│   └── multiprocessing_model.py
├── notebooks/
│   └── Credit_Card_Fraud_Detection.ipynb
├── results/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── time_distribution.png
│   ├── correlation_heatmap.png
│   └── amount_by_class_boxplot.png
├── models/
├── README.md
├── requirements.txt
└── .gitignore
```

### Folders and Files

- **data/**: Contains the original CSV dataset (can be downloaded from "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud").
- **database/**: Contains the SQLite database used for storing the dataset.
- **src/**: All source code files.
  - `save_to_sql.py`: Saves the CSV file into an SQLite database.
  - `data_exploration.py`: Loads data from the database and performs exploratory data analysis (EDA).
  - `data_visualization.py`: Generates visualizations for data analysis.
  - `data_preprocessing.py`: Preprocesses the data (e.g., scaling, train-test split).
  - `model_development.py`: Develops and evaluates a machine learning model.
  - `multiprocessing_model.py`: Uses multiprocessing to perform cross-validation.
- **notebooks/**: Jupyter Notebooks for interactive data analysis.
- **results/**: Stores plots generated during EDA and visualization.
- **models/**: Directory to save trained models.
- **README.md**: Documentation of the project.
- **requirements.txt**: List of dependencies.
- **.gitignore**: Specifies files and folders to ignore in version control.

## Requirements

To run the project, you need to install the following dependencies:

```
$ pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
plotly
sqlalchemy
joblib
```

## Dataset

The dataset used is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) available on Kaggle. It contains transactions made by European cardholders in September 2013, with 492 frauds out of 284,807 transactions.

- **Class 0**: Non-fraudulent transactions
- **Class 1**: Fraudulent transactions

## Steps to Run the Project

1. **Set Up the Environment**
   - Create a virtual environment (optional but recommended):
     ```
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install dependencies:
     ```
     pip install -r requirements.txt
     ```

2. **Load Data into Database**
   - Run the script to load the CSV data into the SQLite database:
     ```
     python src/save_to_sql.py
     ```

3. **Perform Data Exploration**
   - Generate initial exploration reports and plots:
     ```
     python src/data_exploration.py
     ```

4. **Visualize Data**
   - Generate more detailed visualizations:
     ```
     python src/data_visualization.py
     ```

5. **Preprocess Data**
   - Prepare the data for model development:
     ```
     python src/data_preprocessing.py
     ```

6. **Develop and Evaluate Model**
   - Train a logistic regression model and evaluate its performance:
     ```
     python src/model_development.py
     ```

7. **Run Multiprocessing Model**
   - Use multiprocessing to speed up cross-validation:
     ```
     python src/multiprocessing_model.py
     ```

## Results

- **Class Distribution**: Most transactions are non-fraudulent, leading to a highly imbalanced dataset.
- **EDA Plots**: Various plots are saved in the `results/` folder, including:
  - `class_distribution.png`: Visualizes the class imbalance.
  - `amount_distribution.png`: Shows the distribution of transaction amounts.
  - `time_distribution.png`: Displays the distribution of transaction times.
  - `correlation_heatmap.png`: Shows correlations between different features.
  - `amount_by_class_boxplot.png`: Boxplot of transaction amounts by class.

## Model Performance

- The average F1 score obtained from cross-validation is around **0.727**. Given the class imbalance, this metric is crucial to evaluate how well the model balances precision and recall.

## Improving the Model

- **Handling Class Imbalance**: Future improvements could include techniques like SMOTE (Synthetic Minority Over-sampling Technique), undersampling, or using algorithms designed to handle imbalance, such as XGBoost.
- **Feature Engineering**: Creating additional features may help the model distinguish between fraudulent and non-fraudulent transactions.
- **Hyperparameter Tuning**: Techniques like GridSearchCV or RandomizedSearchCV could be used to improve model performance.

## License

This project is licensed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

- The dataset is provided by Worldline and the Machine Learning Group (MLG) at ULB (Université Libre de Bruxelles) and is available on Kaggle.
- Inspiration from the open datasets community and the challenges of working with imbalanced data.

## Contact

For any questions or suggestions, please reach out to [your-email@example.com].

