# model_evaluation.py (Matches CampusX Video behavior - No Schema)

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
import shutil


# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "dpbd-2004"
repo_name = "mlops-mini-project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# --- CONFIGURATION ---
# mlflow.set_tracking_uri('https://dagshub.com/dpbd-2004/mlops-mini-project.mlflow')
# dagshub.init(repo_owner='dpbd-2004', repo_name='mlops-mini-project', mlflow=True)

# --- LOGGING SETUP ---
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

def load_data(file_path: str):
    """Load data from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise e

def evaluate_model(clf, X_test, y_test):
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise e

def save_json(data, path):
    """Save dictionary to JSON."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    mlflow.set_experiment("dvc-pipeline")
    
    with mlflow.start_run() as run:
        try:
            logger.debug("Loading data and model...")
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')
            
            # Prepare Data (Numpy)
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            # Evaluate
            metrics = evaluate_model(clf, X_test, y_test)
            save_json(metrics, 'reports/metrics.json')

            # Log Metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
                
            # Log Params
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for k, v in params.items():
                    mlflow.log_param(k, v)

            # --- ROBUST MODEL UPLOAD (No Signature) ---
            local_model_path = "model_to_upload"
            if os.path.exists(local_model_path):
                shutil.rmtree(local_model_path)
            
            logger.debug(f"Saving model locally to {local_model_path}...")
            
            # REMOVED: signature=signature (Matches CampusX)
            mlflow.sklearn.save_model(clf, local_model_path)
            
            logger.debug("Uploading model artifacts...")
            mlflow.log_artifacts(local_model_path, artifact_path="model")
            
            # Clean up
            if os.path.exists(local_model_path):
                shutil.rmtree(local_model_path)
            
            logger.debug("Model uploaded successfully!")

            # Save Info for Registration
            save_json({
                'run_id': run.info.run_id,
                'model_path': 'model'
            }, 'reports/experiment_info.json')
            
            # Log Helper Artifacts
            mlflow.log_artifact('reports/experiment_info.json')
            mlflow.log_artifact('reports/metrics.json')
            mlflow.log_artifact('model_evaluation_errors.log')
            
        except Exception as e:
            logger.error(f"FAILED: {e}")
            print(f"CRITICAL FAILURE: {e}")
            raise e

if __name__ == '__main__':
    main()