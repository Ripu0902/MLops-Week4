# MLOps — Week 7 Graded Assignment

**Student:** Ripunjay Kumar  
**Roll No:** 21f3002511  
**Term:** Sept 2025  
**Course:** MLOps (BSDA5014)


# Iris Classification MLOps Pipeline

A production-ready MLOps pipeline for Iris species classification using Decision Trees, featuring MLflow experiment tracking, DVC data versioning, FastAPI serving, and Kubernetes deployment on Google Kubernetes Engine (GKE).

## Overview

This project demonstrates a complete machine learning operations workflow:

- **Model Training**: Decision Tree classifier with MLflow experiment tracking
- **Data Versioning**: DVC integration with Google Cloud Storage
- **Model Registry**: MLflow model versioning and registry
- **API Service**: FastAPI with health checks, structured logging, and OpenTelemetry tracing
- **Containerization**: Docker image for reproducible deployments
- **Orchestration**: Kubernetes manifests with HPA (Horizontal Pod Autoscaler)
- **CI/CD**: GitHub Actions for automated testing, evaluation, and deployment
- **Load Testing**: wrk-based performance testing with detailed metrics

## Project Structure

```
.
├── app/
│   └── main.py                          # FastAPI application with prediction endpoint
├── data/
│   ├── data.csv.dvc                     # DVC-tracked dataset
│   └── .gitignore                       # Ignore actual data files
├── gke-deploy/
│   ├── deployment.yaml                  # Kubernetes Deployment manifest
│   ├── service.yaml                     # LoadBalancer Service
│   └── hpa.yaml                         # Horizontal Pod Autoscaler
├── src/
│   └── evaluate_latest_model_mlflow.py  # Model evaluation script
├── tests/
│   └── test_evaluation_1.py             # Pytest with Pandera validation
├── train_iris_dt_mlflow.py              # Training script with MLflow logging
├── iris-api-test.lua                    # wrk load testing script
├── Dockerfile                           # Container image definition
├── req.txt                              # Full development dependencies
├── req-app.txt                          # Minimal API dependencies
└── README.md                            # This file
```

## Prerequisites

- Python 3.10+
- Docker
- kubectl (for Kubernetes deployment)
- Google Cloud SDK (for GKE deployment)
- MLflow tracking server
- DVC with Google Cloud Storage backend

## Setup

### 1. Environment Setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r req.txt
```

### 2. Configure Environment Variables

Create a `.env` file or set environment variables:

```powershell
# MLflow tracking server URI
setx MLFLOW_TRACKING_SERVER "http://your-mlflow-server:6969"
```

### 3. Pull Data from DVC

```powershell
dvc pull data/data.csv.dvc
```

## Usage

### Training

Train the Decision Tree model with MLflow experiment tracking:

```powershell
python train_iris_dt_mlflow.py
```

This script:
- Pulls the latest data from DVC
- Trains multiple Decision Tree configurations
- Logs parameters, metrics, and artifacts to MLflow
- Registers the best model to MLflow Model Registry as `IrisDecisionTreeModel`

### Evaluation

Evaluate the latest registered model:

```powershell
python src/evaluate_latest_model_mlflow.py
```

### Testing

Run automated tests with data validation:

```powershell
pytest -q
```

The test suite includes:
- Pandera schema validation for data quality
- Model evaluation from GCS artifacts
- Confusion matrix generation
- CML report creation for CI/CD

### Local API Development

Start the FastAPI server locally:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API endpoints:
- `GET /` - Health check
- `GET /live` - Liveness probe
- `GET /ready` - Readiness probe (checks model loaded)
- `POST /predict` - Make predictions

Example prediction request:

```powershell
curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## Docker

### Build Image

```powershell
docker build -t iris-api -f Dockerfile .
```

### Run Container

```powershell
docker run -p 8000:8000 `
  --env MLFLOW_TRACKING_SERVER="http://your-mlflow-server:6969" `
  iris-api
```

## Kubernetes Deployment

### Prerequisites

1. GKE cluster with Workload Identity enabled
2. Artifact Registry repository for Docker images
3. Service account with appropriate IAM roles

### Deploy to GKE

```bash
# Apply Kubernetes manifests
kubectl apply -f gke-deploy/deployment.yaml
kubectl apply -f gke-deploy/service.yaml
kubectl apply -f gke-deploy/hpa.yaml

# Monitor deployment
kubectl rollout status deployment/iris-api-deployment
kubectl get pods -l app=iris-api -w

# Get external IP
kubectl get service iris-api-service -w
```

### Kubernetes Resources

**Deployment** (`deployment.yaml`):
- Initial replicas: 1
- Container: FastAPI app with MLflow model
- Health checks: Liveness and readiness probes
- Resource limits: 1Gi memory, 500m CPU
- Service account: `iris-api-ksa` (Workload Identity)

**Service** (`service.yaml`):
- Type: LoadBalancer
- External port: 80
- Target port: 8000

**HPA** (`hpa.yaml`):
- Min replicas: 1
- Max replicas: 3
- CPU target: 80%
- Memory target: 70%

## Load Testing

Test API performance using wrk:

```bash
# Basic load (2 threads, 10 connections, 10 seconds)
wrk -t2 -c10 -d10s -s iris-api-test.lua http://YOUR_EXTERNAL_IP:80

# Intermediate load (4 threads, 50 connections, 30 seconds)
wrk -t4 -c50 -d30s -s iris-api-test.lua http://YOUR_EXTERNAL_IP:80

# Heavy load (8 threads, 100 connections, 60 seconds)
wrk -t8 -c100 -d60s -s iris-api-test.lua http://YOUR_EXTERNAL_IP:80
```

The load test script (`iris-api-test.lua`) provides detailed metrics:
- Total requests and success rate
- Requests per second
- Latency statistics (min, max, mean, percentiles)
- Error breakdown by type

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/`) automates:

1. **Testing**: Run pytest with data validation
2. **Evaluation**: Load latest model from GCS and evaluate
3. **Reporting**: Generate confusion matrix and metrics
4. **Build**: Create Docker image
5. **Push**: Upload to Artifact Registry
6. **Deploy**: Update Kubernetes deployment
7. **Load Test**: Run wrk tests and record results
8. **Artifacts**: Upload test results and reports

## Key Features

### MLflow Integration
- Experiment tracking with parameters and metrics
- Model registry for version management
- Artifact storage in Google Cloud Storage
- Automatic model selection based on accuracy

### Data Versioning
- DVC tracks dataset versions
- Remote storage in GCS
- Git integration for reproducibility

### API Features
- FastAPI with automatic OpenAPI documentation
- Pydantic validation for request/response
- Structured JSON logging
- OpenTelemetry distributed tracing
- Health and readiness probes
- Process time headers

### Kubernetes Features
- Horizontal Pod Autoscaling based on CPU/memory
- Rolling updates with zero downtime
- Resource requests and limits
- Workload Identity for secure GCP access
- LoadBalancer for external access

## Monitoring & Observability

The API includes comprehensive observability:

- **Structured Logging**: JSON-formatted logs with trace IDs
- **Distributed Tracing**: OpenTelemetry with Cloud Trace export
- **Health Checks**: Liveness and readiness endpoints
- **Metrics**: Request duration, status codes, predictions

## Security Best Practices

- Environment variables for sensitive configuration
- Workload Identity instead of service account keys
- Resource limits to prevent resource exhaustion
- Health checks for automatic recovery
- No hardcoded credentials in code or manifests

## Troubleshooting

### Model Not Loading
- Verify `MLFLOW_TRACKING_SERVER` environment variable
- Check MLflow registry has `IrisDecisionTreeModel` version 1
- Review pod logs: `kubectl logs -l app=iris-api`

### Pod Not Ready
- Check readiness probe: `kubectl describe pod <pod-name>`
- Verify model loads successfully in logs
- Ensure MLflow server is accessible from cluster

### HPA Not Scaling
- Verify metrics-server is installed: `kubectl get deployment metrics-server -n kube-system`
- Check HPA status: `kubectl get hpa iris-api-hpa`
- Review resource utilization: `kubectl top pods`

## Development Notes

- The project uses Python 3.10 for compatibility with all dependencies
- `req.txt` contains full development dependencies
- `req-app.txt` is minimal for production container images
- DVC remote must be configured before `dvc pull`
- MLflow tracking server must be running before training/serving

## License

This project is part of an MLOps course assignment.

## Contact

For issues or questions, please open an issue in this repository.
