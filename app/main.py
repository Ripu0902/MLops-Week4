import logging
import time
import pandas as pd
import os
import json
from dotenv import load_dotenv
from typing import List, Optional

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, Field
import mlflow.pyfunc

# OpenTelemetry Imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# --- Setup Tracer ---
# Ensure this runs only once
provider = TracerProvider()
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
provider.add_span_processor(span_processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# --- Setup Structured Logging ---
# We configure the formatter to just output the message.
# We will manually format our messages as JSON strings.
logger = logging.getLogger("demo-log-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s') # Pass through the JSON string
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

load_dotenv()

# --- Pydantic Schemas ---

class IrisFeatures(BaseModel):
    """Input features for the Iris model."""
    sepal_length: float = Field(..., example=5.1, description="Sepal length in cm")
    sepal_width: float = Field(..., example=3.5, description="Sepal width in cm")
    petal_length: float = Field(..., example=1.4, description="Petal length in cm")
    petal_width: float = Field(..., example=0.2, description="Petal width in cm")

class PredictionOut(BaseModel):
    """Output schema for the prediction."""
    prediction: List[int] = Field(..., example=[0], description="List of predicted class labels")

# --- FastAPI App Initialization ---

app = FastAPI(
    title="Iris Decision Tree Classifier API",
    description="API for predicting Iris species using an MLflow-trained model.",
    version="1.0.0"
)

# --- Utility Function for Tracing ---

def get_trace_id() -> Optional[str]:
    """Helper to safely get the current trace ID."""
    span = trace.get_current_span()
    if span.get_span_context().is_valid:
        return format(span.get_span_context().trace_id, "032x")
    return None

# --- Middleware for Logging and Timing ---

@app.middleware("http")
async def add_process_time_and_logging(request: Request, call_next):
    """
    Middleware to log requests, responses, and add process time.
    """
    with tracer.start_as_current_span(f"{request.method} {request.url.path}") as span:
        start_time = time.time()
        trace_id = get_trace_id()
        
        # Log request start
        logger.info(json.dumps({
            "event": "request_start",
            "severity": "INFO",
            "trace_id": trace_id,
            "method": request.method,
            "path": str(request.url),
            "client_host": request.client.host if request.client else "unknown",
        }))

        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)

            # Log request end
            logger.info(json.dumps({
                "event": "request_end",
                "severity": "INFO",
                "trace_id": trace_id,
                "method": request.method,
                "path": str(request.url),
                "status_code": response.status_code,
                "duration_ms": round(process_time * 1000, 2),
            }))
            
            span.set_attribute("http.status_code", response.status_code)
            span.set_attribute("http.process_time_ms", round(process_time * 1000, 2))

        except Exception as exc:
            # This catches exceptions in the request processing
            process_time = time.time() - start_time
            span.set_attribute("http.status_code", 500)
            span.record_exception(exc)
            
            # Log the exception here before it goes to the exception handlers
            logger.error(json.dumps({
                "event": "middleware_exception",
                "severity": "ERROR",
                "trace_id": trace_id,
                "method": request.method,
                "path": str(request.url),
                "duration_ms": round(process_time * 1000, 2),
                "error": str(exc),
            }))
            # Re-raise to be handled by FastAPI's exception handlers
            raise exc

        return response

# --- Exception Handlers ---

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handles Pydantic validation errors (HTTP 422)."""
    trace_id = get_trace_id()
    logger.warning(json.dumps({
        "event": "validation_error",
        "severity": "WARNING",
        "trace_id": trace_id,
        "path": str(request.url),
        "error_details": exc.errors(),
    }))
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation Error",
            "errors": exc.errors(),
            "trace_id": trace_id
        },
    )

# --- Startup Event to Load Model ---

@app.on_event("startup")
def load_model():
    """
    Load the MLflow model into the app state on startup.
    """
    with tracer.start_as_current_span("load_model_startup"):
        app.state.model = None
        try:
            MLFLOW_TRACKING_SERVER = os.environ.get("MLFLOW_TRACKING_SERVER")
            if not MLFLOW_TRACKING_SERVER:
                logger.critical(json.dumps({
                    "event": "model_load_fail",
                    "severity": "CRITICAL",
                    "error": "MLFLOW_TRACKING_SERVER env var not set."
                }))
                return

            MODEL_NAME = "IrisDecisionTreeModel"
            MODEL_VERSION = "1" # Using version 1 as per your original code
            
            mlflow.set_tracking_uri(MLFLOW_TRACKING_SERVER)
            
            logger.info(json.dumps({
                "event": "model_load_start",
                "severity": "INFO",
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "tracking_uri": MLFLOW_TRACKING_SERVER
            }))

            model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
            app.state.model = mlflow.pyfunc.load_model(model_uri=model_uri)
            
            logger.info(json.dumps({
                "event": "model_load_success",
                "severity": "INFO",
                "model_uri": model_uri
            }))

        except Exception as e:
            logger.exception(json.dumps({
                "event": "model_load_fail",
                "severity": "CRITICAL",
                "error": str(e)
            }), exc_info=True)

# --- API Endpoints ---

@app.get("/", tags=["Health"])
def home():
    """A simple endpoint to confirm the API is running."""
    return {"message": "Iris Decision Tree Model API is running... V1"}

@app.get("/live", tags=["Health"])
def live_check():
    """
    Liveness probe. Checks if the application process is running.
    """
    return {"status": "alive"}

@app.get("/ready", tags=["Health"])
def ready_check(response: Response):
    """
    Readiness probe. Checks if the application is ready to serve requests
    (i.e., the model is loaded).
    """
    if app.state.model:
        return {"status": "ready", "model": "loaded"}
    else:
        response.status_code = 503  # Service Unavailable
        return {"status": "not_ready", "model": "not_loaded"}

@app.post("/predict", response_model=PredictionOut, tags=["Prediction"])
def predict(features: IrisFeatures, request: Request):
    """
    Make a prediction on a single set of Iris features.
    
    **Input:**
    - `sepal_length` (float): Sepal length in cm
    - `sepal_width` (float): Sepal width in cm
    - `petal_length` (float): Petal length in cm
    - `petal_width` (float): Petal width in cm
    
    **Output:**
    - `prediction` (List[int]): A list containing the predicted class label (e.g., `[0]`).
    """
    with tracer.start_as_current_span("predict_endpoint") as span:
        trace_id = get_trace_id()
        model = request.app.state.model
        
        if not model:
            logger.error(json.dumps({
                "event": "prediction_fail",
                "severity": "ERROR",
                "trace_id": trace_id,
                "error": "Model is not loaded."
            }))
            raise HTTPException(status_code=503, detail="Model is not loaded. Service is unavailable.")

        try:
            # Convert Pydantic model to DataFrame
            # model_dump() is the modern equivalent of .dict()
            df = pd.DataFrame([features.model_dump()])
            
            # Run prediction
            prediction = model.predict(df)
            prediction_list = prediction.tolist()

            # Log successful prediction
            logger.info(json.dumps({
                "event": "prediction_success",
                "severity": "INFO",
                "trace_id": trace_id,
                "input_features": features.model_dump(),
                "prediction": prediction_list
            }))
            
            # Add prediction as span attribute
            span.set_attribute("ml.prediction", json.dumps(prediction_list))

            return {"prediction": prediction_list}
        
        except Exception as e:
            # This catches errors during the predict() call itself
            logger.exception(json.dumps({
                "event": "prediction_fail",
                "severity": "ERROR",
                "trace_id": trace_id,
                "error": str(e)
            }), exc_info=True)
            raise HTTPException(status_code=500, detail=f"Prediction error: {e}", headers={"X-Trace-ID": trace_id})

# Example of how to run this file (e.g., using uvicorn)
# uvicorn main:app --host 0.0.0.0 --port 8000