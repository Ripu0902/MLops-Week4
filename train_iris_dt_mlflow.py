#!/usr/bin/env python3
"""
Train DecisionTreeClassifier on DVC-tracked Iris dataset and log experiments to MLflow.
Logs dataset version (DVC + Git) for reproducibility.
"""

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import subprocess
import os
from dotenv import load_dotenv
load_dotenv()


# ======================
# CONFIGURATION
# ======================
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_SERVER")  # MLflow tracking server URI
EXPERIMENT_NAME = "iris_decision_tree"
DATA_CSV = "data/data.csv"             # local CSV path after dvc pull
DATA_DVC_FILE= "data/data.csv.dvc"

# ======================
# SETUP MLflow
# ======================
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
client = mlflow.tracking.MlflowClient()


# ======================
# PULL DATA FROM DVC
# ======================
print(f"Pulling dataset from DVC remote: {DATA_DVC_FILE} ...")
subprocess.run(["dvc", "pull", DATA_DVC_FILE], check=True)
print("Dataset pulled successfully.")


# ======================
# LOAD DATA
# ======================
df = pd.read_csv(DATA_CSV)
X = df.drop("species", axis=1).values
y = df["species"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# LOG DATA VERSION INFO
# ======================
git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
mlflow.set_tag("git_commit", git_commit)

# Optionally, log DVC status for dataset
dvc_status = subprocess.check_output(["dvc", "status", DATA_DVC_FILE]).decode().strip()
mlflow.set_tag("dvc_status", dvc_status)

# ======================
# EXPERIMENT CONFIGURATIONS
# ======================
param_grid = [
    {"max_depth": 2, "criterion": "gini"},
    {"max_depth": 3, "criterion": "entropy"},
]

if mlflow.active_run() is not None:
    mlflow.end_run()


# ======================
# TRAIN AND LOG EXPERIMENTS
# ======================
for params in param_grid:
    with mlflow.start_run(run_name=f"dt_maxdepth_{params['max_depth']}"):
        # --- Model training ---
        clf = DecisionTreeClassifier(**params, random_state=42)
        clf.fit(X_train, y_train)

        # --- Predictions & Metrics ---
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        # --- Log Parameters & Metrics ---
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_param("criterion", params["criterion"])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # --- Infer model signature ---
        signature = infer_signature(X_train, clf.predict(X_train))

        # --- Log model artifact with signature ---
        mlflow.sklearn.log_model(
            clf,
            name="model",
            signature=signature,
            input_example=X_train[:5]  # optional example input for serving
        )

        print(
            f"Run complete: depth={params['max_depth']}, "
            f"criterion={params['criterion']}, acc={acc:.4f}, f1={f1:.4f}"
        )

# ======================
# REGISTER BEST MODEL
# ======================
runs = client.search_runs(
    experiment_ids=[client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=4,
)

if not runs:
    raise RuntimeError("No runs found for this experiment!")

# Find the highest accuracy
best_accuracy = max(r.data.metrics.get("accuracy", 0) for r in runs)

# Get all runs that have this best accuracy
best_runs = [r for r in runs if r.data.metrics.get("accuracy", 0) == best_accuracy]

# Choose the most recent run among tied best runs
best_run = max(best_runs, key=lambda r: r.info.start_time)

print(f"Best Run : {best_run}")
best_run_id = best_run.info.run_id
best_model_uri = f"runs:/{best_run_id}/model"

print(f"Best Run ID: {best_run_id}")
print(f"Best Accuracy: {best_accuracy}")
print(f"Best Model URI: {best_model_uri}")

model_name = "IrisDecisionTreeModel"
mlflow.register_model(model_uri=best_model_uri, name=model_name)

print(f" Best model registered to MLflow Model Registry as '{model_name}'")
