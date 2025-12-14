# src/data_prep.py

import os
import shutil
import json
import datetime
import pandas as pd
import numpy as np
import warnings
from pprint import pprint

# Scikit-learn for transformation
from sklearn.preprocessing import MinMaxScaler

# Filter warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: "%.3f" % x)

# --- Define Helper Functions ---

def describe_numeric_col(x):
    """
    Calculates various descriptive stats for a numeric column in a dataframe.
    """
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )

def impute_missing_values(x, method="mean"):
    """
    Imputes the mean/median for numeric columns or the mode for other types.
    """
    if (x.dtype == "float64") or (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method=="mean" else x.fillna(x.median())
    else:
        # Impute mode, handling edge case where mode() returns multiple values
        x = x.fillna(x.mode().iloc[0])
    return x

# --- Define Main Pipeline Logic ---

def run_data_processing(raw_data_path: str, artifacts_dir: str):
    """
    Executes the data cleaning, feature engineering, and standardization steps.
    
    Args:
        raw_data_path: Path to the raw input CSV file (e.g., 'data/raw_data.csv').
        artifacts_dir: Path to the directory where all artifacts will be stored (e.g., 'artifacts').
    """
    print(f"--- Starting Data Processing ---")
    
    # 1. Setup Artifacts Directory
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Date limits extracted from notebook setup
    max_date_str = "2024-01-31"
    min_date_str = "2024-01-01"

    # 2. Read and Filter Data
    print(f"Loading training data from {raw_data_path}")
    data = pd.read_csv(raw_data_path)
    print("Total rows before filtering:", len(data))

    # Apply date filtering logic from the notebook
    max_date = pd.to_datetime(max_date_str).date()
    min_date = pd.to_datetime(min_date_str).date()
    
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]
    
    min_date = data["date_part"].min()
    max_date = data["date_part"].max()
    
    date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
    date_limits_path = os.path.join(artifacts_dir, "date_limits.json")
    with open(date_limits_path, "w") as f:
        json.dump(date_limits, f)
    print(f"Saved date limits to {date_limits_path}")

    # 3. Feature Selection
    data = data.drop(
        [
            "is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen", 
            # Columns explicitly removed in the second step of the notebook:
            "domain", "country", "visited_learn_more_before_booking", "visited_faq"
        ],
        axis=1
    )
    print("Columns after initial drop:", data.shape[1])
    
    # 4. Data Cleaning (Remove rows with empty target/key variables)
    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)
    
    data = data.dropna(axis=0, subset=["lead_indicator", "lead_id"])
    data = data[data.source == "signup"]
    
    print("Total rows after cleaning:", len(data))
    
    # 5. Create Categorical Data Columns and Split
    vars_to_object = [
        "lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"
    ]
    for col in vars_to_object:
        data[col] = data[col].astype("object")

    # Separate categorical and continuous columns
    cont_vars = data.select_dtypes(include=['float64', 'int64'])
    cat_vars = data.select_dtypes(include=['object'])
    
    print("\nContinuous columns:", len(cont_vars.columns))
    print("Categorical columns:", len(cat_vars.columns))

    # 6. Outlier Handling (Clipping)
    # The notebook clips outliers using mean +/- 2*std
    cont_vars = cont_vars.apply(
        lambda x: x.clip(lower = (x.mean()-2*x.std()), upper = (x.mean()+2*x.std()))
    )
    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary_path = os.path.join(artifacts_dir, 'outlier_summary.csv')
    outlier_summary.to_csv(outlier_summary_path)
    print(f"Saved outlier summary to {outlier_summary_path}")

    # 7. Impute Data
    # Impute missing values for categorical and continuous variables
    
    # Continuous variables missing values (impute mean)
    cont_vars = cont_vars.apply(impute_missing_values)
    
    # Categorical variables missing values (special case for customer_code, then general mode imputation)
    cat_vars.loc[cat_vars['customer_code'].isna(), 'customer_code'] = 'None'
    cat_vars = cat_vars.apply(impute_missing_values)
    
    # 8. Data Standardization (MinMaxScaler)
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    scaler = MinMaxScaler()
    
    # Only fit and transform the continuous variables
    scaler.fit(cont_vars)
    
    # Save scaler artifact
    import joblib
    joblib.dump(value=scaler, filename=scaler_path)
    print(f"Saved scaler in {scaler_path}")
    
    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)

    # 9. Combine Data
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)
    print(f"Data cleansed and combined. Rows: {len(data)}")

    # 10. Data Drift Artifact & Initial Training Data Save
    # Save the columns list before feature engineering steps (like binning/dummy creation)
    data_columns = list(data.columns)
    columns_drift_path = os.path.join(artifacts_dir, 'columns_drift.json')
    with open(columns_drift_path, 'w+') as f:           
        json.dump(data_columns, f)
    print(f"Saved columns drift list to {columns_drift_path}")
    
    training_data_path = os.path.join(artifacts_dir, 'training_data.csv')
    data.to_csv(training_data_path, index=False)
    print(f"Saved intermediate training data to {training_data_path}")

    # 11. Binning Object Columns (Feature Engineering)
    # NOTE: The binning logic in the notebook is slightly messy and incomplete,
    # mapping 'source' values to 'bin_source' but then having conflicting 
    # original logic (data.loc[~data['source'].isin(values_list)]).
    # We follow the final mapping logic as it overwrites previous logic.
    
    # First, create the column (based on the original notebook's first step)
    data['bin_source'] = data['source'] 
    
    # Second, apply the final map (which overwrites 'li', 'organic', 'signup', 'fb')
    mapping = {
        'li' : 'socials', 
        'fb' : 'socials', 
        'organic': 'group1', 
        'signup': 'group1'
    }
    data['bin_source'] = data['source'].map(mapping).fillna(data['bin_source'])
    print("Applied binning for 'source' column.")

    # 12. Save Gold Medallion Dataset
    gold_data_path = os.path.join(artifacts_dir, 'train_data_gold.csv')
    data.to_csv(gold_data_path, index=False)
    print(f"Saved final golden training data to {gold_data_path}")

    print(f"--- Data Processing Complete ---")


# --- Execution Entry Point ---

if __name__ == "__main__":
    # Define paths based on your MLOps structure
    RAW_DATA_INPUT_PATH = os.path.join("data", "raw_data.csv") 
    ARTIFACTS_OUTPUT_DIR = "artifacts"
    
    # Check if raw data exists at the new path
    if not os.path.exists(RAW_DATA_INPUT_PATH):
        raise FileNotFoundError(
            f"Required raw data not found at: {RAW_DATA_INPUT_PATH}. "
            "Please ensure you have moved the file here."
        )

    run_data_processing(RAW_DATA_INPUT_PATH, ARTIFACTS_OUTPUT_DIR)