import os
import json
import pytest
import dvc.api
import pandera as pa
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import storage

# =============== CONFIG ==================
DATA_PATH = "data/data.csv"
REPORT_PATH = "cml_report.md"
CONF_MATRIX_PATH = "confusion_matrix.png"
METRICS_PATH = "metrics.json"
# Point to the root bucket/folder containing run artifacts
GCS_BUCKET = "gs://vertex-mlflow-artifacts-electric-wave-472614-d5"

# ----- Pandera Schema -----
iris_schema = pa.DataFrameSchema({
    "sepal_length": pa.Column(float, pa.Check.in_range(4.0, 8.0)),
    "sepal_width": pa.Column(float, pa.Check.in_range(2.0, 5.0)),
    "petal_length": pa.Column(float, pa.Check.in_range(1.0, 7.0)),
    "petal_width": pa.Column(float, pa.Check.in_range(0.0, 3.0)),
    "species": pa.Column(str, pa.Check.isin(['setosa', 'versicolor', 'virginica']))
})

@pytest.fixture(scope="module")
def dvc_data():
    """Fetch data tracked by DVC."""
    with dvc.api.open(DATA_PATH, mode="r", encoding="utf-8") as f:
        df = pd.read_csv(f)
    return df

def test_data_validation(dvc_data):
    """Validate dataset structure using Pandera."""
    print("\n🔍 Validating data schema...")
    try:
        iris_schema.validate(dvc_data, lazy=True)
        print("✅ Data validation passed!")
    except pa.errors.SchemaErrors as err:
        pytest.fail(f"❌ Data validation failed:\n{err}")

# ---- Load latest model directly from GCS using MLflow ----
def get_latest_model_from_gcs(bucket_uri):
    """
    Fetch the latest MLflow model artifact folder from GCS.
    """
    print(f"Fetching latest MLflow model from {bucket_uri} ...")

    client = storage.Client()
    bucket_name = bucket_uri.replace("gs://", "").split("/")[0]
    prefix = "/".join(bucket_uri.replace("gs://", "").split("/")[1:])

    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    # Filter only MLflow model artifact folders
    model_folders = [b.name for b in blobs if b.name.endswith("/model/") or "model" in b.name.lower()]
    if not model_folders:
        raise FileNotFoundError(f"No model folder found under {bucket_uri}")

    latest_model_path = max(model_folders)
    model_uri = f"gs://{bucket_name}/{latest_model_path}"
    print(f"✅ Latest model path: {model_uri}")

    # Load using MLflow
    model = mlflow.sklearn.load_model(model_uri)
    return model

def test_model_evaluation(dvc_data):
    """Evaluate latest model from GCS and generate report."""
    print("\n🚀 Evaluating latest model...")
    model = get_latest_model_from_gcs(GCS_BUCKET)

    X = dvc_data.drop("species", axis=1).values
    y_true = dvc_data["species"].values

    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    print(f"🎯 Evaluation complete — Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # ---- Confusion Matrix ----
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — Iris Classifier")
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_PATH)
    print(f"📊 Saved confusion matrix to {CONF_MATRIX_PATH}")

    # ---- CML Markdown Report ----
    with open(REPORT_PATH, "w") as f:
        f.write("## Model Evaluation Report\n")
        f.write(f"- Accuracy: `{acc:.4f}`\n")
        f.write(f"- F1 Score: `{f1:.4f}`\n\n")
        f.write("### Confusion Matrix\n")
        f.write(f"![Confusion Matrix]({CONF_MATRIX_PATH})\n")

    # ---- Metrics JSON ----
    metrics = {"accuracy": acc, "f1_score": f1}
    with open(METRICS_PATH, "w") as jf:
        json.dump(metrics, jf)

    # ---- Assertions for CI/CD ----
    assert acc > 0.8, f"Model accuracy too low: {acc}"
    assert f1 > 0.8, f"Model F1 score too low: {f1}"

    print("✅ Evaluation and reporting completed successfully.")
