import os
import json
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
load_dotenv()

# ============= CONFIG =================
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_SERVER")  
DATA_PATH = "data/data.csv"
# ======================================

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()
print(f" Using MLflow tracking URI: {TRACKING_URI}")


def get_latest_registered_model(client: MlflowClient):
    """Return (model_name, model_version) of the most recently registered model."""
    all_models = client.search_registered_models()
    if not all_models:
        raise RuntimeError(" No registered models found in MLflow Registry!")

    # Flatten all versions from all models
    all_versions = []
    for m in all_models:
        for v in m.latest_versions:
            all_versions.append(v)

    # Pick the one with the most recent creation timestamp
    latest = max(all_versions, key=lambda v: int(v.creation_timestamp))
    return latest.name, latest.version


# ---- Find latest model automatically ----
model_name, model_version = get_latest_registered_model(client)
model_uri = f"models:/{model_name}/{model_version}"

print(f" Using latest registered model: {model_name} (version {model_version})")
print(f"Model URI: {model_uri}")

# ---- Load model ----
model = mlflow.sklearn.load_model(model_uri)

# ---- Load evaluation data ----
df = pd.read_csv(DATA_PATH)
X = df.drop("species", axis=1).values
y_true = df["species"].values

# ---- Evaluate ----
y_pred = model.predict(X)
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")
cm = confusion_matrix(y_true, y_pred)

print(f"🎯 Evaluation complete — Accuracy: {acc:.4f}, F1: {f1:.4f}")
