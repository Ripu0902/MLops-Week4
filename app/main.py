from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Iris Decision Tree Classifier API")

# Load model from MLflow Model Registry
MLFLOW_TRACKING_SERVER = os.environ.get("MLFLOW_TRACKING_SERVER")
MODEL_NAME = "IrisDecisionTreeModel"
MODEL_STAGE = "None"  # since you registered once, not promoted yet

mlflow.set_tracking_uri(MLFLOW_TRACKING_SERVER)

# load the latest version of the registered model
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/1")

@app.get("/")
def home():
    return {"message": "Iris Decision Tree Model API is running"}

@app.post("/predict")
def predict(features: dict):
    """
    Example input:
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }
    """
    if model:
        df = pd.DataFrame([features])
        prediction = model.predict(df)
        return {"prediction": prediction.tolist()}
    else:
        return {"Error" : "Model is not loaded"}