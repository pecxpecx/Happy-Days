# src/train.py

import os
import json
import datetime
import pandas as pd
import numpy as np
from pprint import pprint
import joblib

# ML/MLOps libraries
import mlflow
import mlflow.pyfunc
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRFClassifier
from scipy.stats import uniform, randint


# --- Configuration and Constants ---

# Constants used in the notebook (MLflow run name based on current date)
CURRENT_DATE = datetime.datetime.now().strftime("%Y_%B_%d")
DATA_VERSION = "00000"
EXPERIMENT_NAME = CURRENT_DATE
DATA_GOLD_PATH = os.path.join("artifacts", "train_data_gold.csv")
ARTIFACTS_DIR = "artifacts"
MLFLOW_RUNS_DIR = "mlruns" # MLflow default local tracking directory


# --- Define Helper Functions ---

def create_dummy_cols(df, col):
    """
    Creates one-hot encoding columns in the data for a given categorical column.
    """
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df

class lr_wrapper(mlflow.pyfunc.PythonModel):
    """
    Custom wrapper for the Logistic Regression model to log probability predictions 
    to MLflow, matching the notebook's logic.
    """
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        # Predicts probability of the positive class (index 1)
        return self.model.predict_proba(model_input)[:, 1]


# --- Main Training Pipeline Logic ---

def run_training_pipeline(data_gold_path: str, artifacts_dir: str):
    """
    Loads data, trains XGBoost and Logistic Regression models, logs to MLflow, 
    and saves model results.
    """
    print(f"--- Starting Model Training ---")

    # 1. Setup Directories
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(MLFLOW_RUNS_DIR, exist_ok=True)
    # The notebook also creates mlruns/.trash, we skip it as it's not strictly necessary.

    # 2. MLflow Experiment Setup
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)

    # 3. Load Training Data and Initial Split
    data = pd.read_csv(data_gold_path)
    print(f"Training data length: {len(data)}")

    # Drop columns not needed for modeling (as per notebook)
    data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)
    
    # Identify categorical and other columns (as per notebook)
    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    cat_vars = data[cat_cols]
    other_vars = data.drop(cat_cols, axis=1)

    # 4. Dummy Variable Creation (One-Hot Encoding)
    for col in cat_vars:
        # Convert to category first, then create dummies (as per notebook)
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)
    
    # Combine data and ensure all columns are float64
    data = pd.concat([other_vars, cat_vars], axis=1)
    for col in data.columns:
        data[col] = data[col].astype("float64")

    # Final feature/target split
    y = data["lead_indicator"]
    X = data.drop(["lead_indicator"], axis=1)
    
    # Data splitting (using stratify and random_state as defined in notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.15, stratify=y
    )

    # Dictionary to store results for model selection step
    model_results = {}

    # --- 5. Model Training: XGBoost ---
    print("\n--- Training XGBoost Model ---")
    
    model = XGBRFClassifier(random_state=42)
    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"]
    }
    
    # NOTE: Notebook uses cv=10, n_iter=10
    model_grid_xgboost = RandomizedSearchCV(model, param_distributions=params, 
                                            n_jobs=-1, verbose=0, n_iter=10, cv=10)
    model_grid_xgboost.fit(X_train, y_train)

    # Prediction and Metrics
    y_pred_train_xgboost = model_grid_xgboost.predict(X_train)
    y_pred_test_xgboost = model_grid_xgboost.predict(X_test)
    
    print("XGBoost Accuracy test:", accuracy_score(y_pred_test_xgboost, y_test))

    # Save Best XGBoost Model
    xgboost_model = model_grid_xgboost.best_estimator_
    xgboost_model_path = os.path.join(artifacts_dir, "lead_model_xgboost.json")
    xgboost_model.save_model(xgboost_model_path)
    print(f"Saved XGBoost model to {xgboost_model_path}")

    # Store classification report for model selection
    model_results[xgboost_model_path] = classification_report(y_test, y_pred_test_xgboost, output_dict=True)

    # --- 6. Model Training: SKLearn Logistic Regression (Logged with MLflow) ---
    print("\n--- Training Logistic Regression Model (with MLflow) ---")
    
    experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    
    # Start MLflow Run for LR model
    with mlflow.start_run(experiment_id=experiment_id) as run:
        
        lr_model = LogisticRegression()
        lr_model_path = os.path.join(artifacts_dir, "lead_model_lr.pkl")

        # Hyperparameter search space
        lr_params = {
                  'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                  'penalty':  ["none", "l1", "l2", "elasticnet"],
                  'C' : [100, 10, 1.0, 0.1, 0.01]
        }
        
        # NOTE: Notebook uses cv=3, n_iter=10
        model_grid_lr = RandomizedSearchCV(lr_model, param_distributions=lr_params, 
                                           verbose=0, n_iter=10, cv=3)
        model_grid_lr.fit(X_train, y_train)

        best_model_lr = model_grid_lr.best_estimator_

        # Prediction and Metrics
        y_pred_test_lr = model_grid_lr.predict(X_test)
        
        print("Logistic Regression Accuracy test:", accuracy_score(y_pred_test_lr, y_test))

        # Log artifacts and metrics
        mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test_lr, average='weighted'))
        mlflow.log_param("data_version", DATA_VERSION)
        
        # Store model to disk for interpretability
        joblib.dump(value=best_model_lr, filename=lr_model_path)
        
        # Custom python model for predicting probability (as required by notebook)
        mlflow.pyfunc.log_model('model', python_model=lr_wrapper(best_model_lr))
        mlflow.log_artifacts(artifacts_dir, artifact_path="model")
        
        run_id = run.info.run_id # Capture run_id for model selection step
        
    model_classification_report = classification_report(y_test, y_pred_test_lr, output_dict=True)
    model_results[lr_model_path] = model_classification_report

    # 7. Save Columns List and Model Results
    column_list_path = os.path.join(artifacts_dir, 'columns_list.json')
    columns = {'column_names': list(X_train.columns)}
    with open(column_list_path, 'w+') as columns_file:
        json.dump(columns, columns_file)
    print(f'Saved column list to {column_list_path}')

    model_results_path = os.path.join(artifacts_dir, "model_results.json")
    with open(model_results_path, 'w+') as results_file:
        json.dump(model_results, results_file, indent=4)
    print(f'Saved model results to {model_results_path}')
    
    print(f"MLflow Run ID for Logistic Regression: {run_id}")
    print(f"--- Model Training Complete ---")


# --- Execution Entry Point ---

if __name__ == "__main__":
    
    if not os.path.exists(DATA_GOLD_PATH):
        raise FileNotFoundError(
            f"Required golden data not found at: {DATA_GOLD_PATH}. "
            "Please ensure src/data_prep.py has been run successfully."
        )

    run_training_pipeline(DATA_GOLD_PATH, ARTIFACTS_DIR)