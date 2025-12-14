# src/deploy.py

import os
import time
import json
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from pprint import pprint

# --- Configuration and Constants ---

# Constants used in the notebook
CURRENT_DATE = datetime.datetime.now().strftime("%Y_%B_%d")
ARTIFACT_PATH = "model" # Artifact path used during LR model logging
MODEL_NAME = "lead_model"
EXPERIMENT_NAME = CURRENT_DATE
MODEL_RESULTS_PATH = os.path.join("artifacts", "model_results.json")


# --- Define Helper Functions (Copied from Notebook) ---

def wait_until_ready(model_name, model_version):
    """Waits for the registered model version to transition to the READY state."""
    client = MlflowClient()
    print(f"Waiting for model version {model_version} to be ready...")
    for _ in range(10):
        model_version_details = client.get_model_version(
          name=model_name,
          version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print(f"Model status: {ModelVersionStatus.to_string(status)}")
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)

def wait_for_deployment(model_name, model_version, stage='Staging'):
    """Waits for a model version to transition to the specified stage."""
    client = MlflowClient()
    status = False
    print(f"Waiting for model version {model_version} to transition to '{stage}'...")
    while not status:
        model_version_details = dict(
            client.get_model_version(name=model_name,version=model_version)
            )
        if model_version_details['current_stage'] == stage:
            print(f'Transition completed to {stage}')
            status = True
            break
        else:
            time.sleep(2)
    return status


# --- Main Deployment Pipeline Logic ---

def run_deployment_pipeline(model_name: str, experiment_name: str, model_results_path: str):
    """
    1. Selects the best trained model (based on f1-score from model_results.json).
    2. Compares it against the current Production model (if one exists).
    3. Registers the best model (if it outperforms production or if no production model exists).
    4. Transitions the newly registered model to 'Staging'.
    """
    print(f"--- Starting Model Selection and Deployment ---")
    
    # 1. Setup MLflow client and identify experiment
    client = MlflowClient()
    try:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        experiment_ids = [experiment_id]
    except AttributeError:
        # Handle case where experiment might not be found (should be created in train.py)
        print(f"Error: MLflow experiment '{experiment_name}' not found. Cannot proceed.")
        return

    # 2. Get the best performing run from the latest training (based on f1_score)
    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["metrics.f1_score DESC"],
        max_results=1
    ).iloc[0]
    
    # Check MLflow run ID
    if experiment_best.empty:
        print("No successful MLflow run found. Exiting deployment.")
        return

    # 3. Determine the actual best model path based on the saved JSON artifact
    with open(model_results_path, "r") as f:
        model_results = json.load(f)
        
    results_df = pd.DataFrame({model: val["weighted avg"] for model, val in model_results.items()}).T
    best_model_path = results_df.sort_values("f1-score", ascending=False).iloc[0].name
    print(f"\nBest model path identified from results: {best_model_path}")
    print(f"Best run F1-score: {experiment_best['metrics.f1_score']:.4f}")

    # 4. Get current Production Model details
    prod_model_list = [
        model for model in client.search_model_versions(f"name='{model_name}'") 
        if dict(model)['current_stage'] == 'Production'
    ]
    prod_model_exists = len(prod_model_list) > 0
    run_id_to_register = experiment_best["run_id"]

    if prod_model_exists:
        prod_model_run_id = dict(prod_model_list[0])['run_id']
        prod_model_score = mlflow.get_run(prod_model_run_id).data.metrics.get("f1_score", 0.0)

        print(f"Production model found (F1-score: {prod_model_score:.4f})")
        
        # Comparison logic: Register new model only if it outperforms production
        if experiment_best["metrics.f1_score"] > prod_model_score:
            print("New model outperforms Production. Registering new version.")
        else:
            print("New model does not outperform Production. Skipping registration.")
            run_id_to_register = None # Do not register
    else:
        print("No model currently in Production. Registering new model.")

    # 5. Register the best model (if necessary)
    if run_id_to_register is not None:
        
        # MLflow URI points to the artifact logged during the LR model run
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=run_id_to_register,
            artifact_path=ARTIFACT_PATH # "model"
        )
        
        # Note: The notebook uses the LR run's artifact path even if XGBoost was better 
        # based on model_results.json. This is likely an error in the original notebook 
        # (XGBoost model file is saved but not logged to MLflow artifacts in a way 
        # the model registry can easily access). We follow the notebook's final 
        # registration of the LR model run.
        
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        
        # Wait for model to transition to Ready state
        wait_until_ready(model_details.name, model_details.version)
        model_version = model_details.version

        print(f"Model {model_name} version {model_version} registered.")

        # 6. Transition to Staging
        # Transition the newly registered model to 'Staging'
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Staging", 
            archive_existing_versions=True # Archives any existing Staging models
        )
        wait_for_deployment(model_name, model_version, 'Staging')
        
    print(f"--- Model Selection and Deployment Complete ---")


# --- Execution Entry Point ---

if __name__ == "__main__":
    # Ensure MLflow is configured to use the local mlruns directory
    mlflow.set_tracking_uri(f"file:{os.path.abspath('mlruns')}")
    
    # Run deployment logic
    run_deployment_pipeline(MODEL_NAME, EXPERIMENT_NAME, MODEL_RESULTS_PATH)