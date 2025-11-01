# MLOps — Week 6 Graded Assignment

**Student:** Ripunjay Kumar  
**Roll No:** 21f3002511  
**Term:** Sept 2025  
**Course:** MLOps (BSDA5014)

This repository contains the Week 6 graded assignment: an end-to-end MLOps pipeline for Iris classification using MLflow, DVC, automated testing, and CI/CD reporting.

Reproducibility contract (short)
- Inputs: DVC-tracked Iris dataset (`data/data.csv`), training code (`train_iris_dt_mlflow.py`), environment variables (notably `MLFLOW_TRACKING_SERVER`) and access to configured remotes (GCS, Cloud SQL) when running full pipeline.
- Outputs: MLflow runs & registered models, model artifacts (in object storage), evaluation reports (`cml_report.md`, `metrics.json`), and optional Docker image for serving the model.
- Success criteria: training completes and registers a model; evaluation tests assert accuracy and F1 > 0.8; CI produces a markdown report and uploads artifacts.

Prerequisites (local)
- Python 3.10+ installed
- Docker (optional, to build/run the API image)
- DVC (if you plan to pull data from remotes)
- Google Cloud SDK (if using GCS/DVC remotes) and proper service account permissions for CI

Note: the README sections below include quick, copyable PowerShell commands for local setup on Windows. When running in CI or cloud, use repository secrets to supply credentials (do NOT hard-code them).

---

## Project at a glance

- Training: `train_iris_dt_mlflow.py` — Decision Tree experiments logged to MLflow
- Evaluation: `src/evaluate_latest_model_mlflow.py` — evaluates the latest registered model
- API: `app/main.py` — FastAPI server that serves the registered model
- Tests: `tests/test_evaluation_1.py` — data schema checks and model evaluation tests
- Data versioning: DVC (data file: `data/data.csv`, tracked via `data/data.csv.dvc`)
- CI/CD: GitHub Actions (workflows live in `.github/workflows/` when present)

## Repository structure

```text
├── app/                          # FastAPI app for serving the model
│   └── main.py
├── data/                         # DVC-tracked data
│   └── data.csv.dvc
├── src/                          # Utility scripts
│   └── evaluate_latest_model_mlflow.py
├── tests/                        # Test suite (pytest)
│   └── test_evaluation_1.py
├── train_iris_dt_mlflow.py       # Training + MLflow logging + model registration
├── server.sh                     # Helper script to start MLflow tracking server
├── req.txt                       # Full Python dependencies
├── req-app.txt                   # Minimal dependencies for running the API
├── Dockerfile                    # Container image for the API
└── README.md

```

## Quick notes for Week 6 (graded)

- This README was updated for the Week 6 graded assignment. Ensure your branch/PR mentions Week 6 as required by the course instructions.
- Tests in `tests/test_evaluation_1.py` assert minimum performance thresholds (accuracy and F1 > 0.8) and perform data validation using Pandera.

## Requirements

Install core dependencies (use `req.txt` for full environment, `req-app.txt` for a minimal API image):

Windows PowerShell example (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r req.txt
```

For running the API in a lightweight container, `req-app.txt` is used in the provided `Dockerfile`.

## Setup & configuration

1. Create a `.env` file (or set environment variables) with the MLflow tracking server URI:

```powershell
# Example (set your own URI)
setx MLFLOW_TRACKING_SERVER "http://localhost:6969"
```

2. If the project uses DVC remote(s), pull the data locally before training/evaluating:

```powershell
dvc pull data/data.csv.dvc
```

3. If you need to start a local MLflow server (optional), update `server.sh` or provide your own MLflow tracking server and artifact store. The repo includes a `server.sh` helper intended for Unix shells; for Windows, replicate the equivalent commands using PowerShell or WSL.

## How to run

Train (logs runs to MLflow server configured via `MLFLOW_TRACKING_SERVER` env var):

```powershell
python train_iris_dt_mlflow.py
```

Run evaluation (loads the latest registered model from the MLflow registry):

```powershell
python src/evaluate_latest_model_mlflow.py
```

Run tests (pytest):

```powershell
pytest -q
```

Start the FastAPI app locally (requires `MLFLOW_TRACKING_SERVER` and model registered):

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or build and run the Docker image (API-only):

```powershell
docker build -t iris-api -f Dockerfile .
docker run -p 8000:8000 --env MLFLOW_TRACKING_SERVER="<your-server>" iris-api
```

## Tests and CI expectations

- The tests validate both dataset schema (Pandera) and model evaluation. Tests assume the latest model artifacts are available in the configured artifact store or registry used by MLflow.
- CI workflows (if present) will:
  - authenticate to GCP using a service account (`GCP_SA_KEY` secret)
  - pull DVC data
  - run tests and generate a CML-like markdown report

## DVC & data

- Data is tracked with DVC (see `data/data.csv.dvc`). Use `dvc pull` to fetch the versioned dataset from the configured remote.
- If you don't have DVC remotes configured locally, the test fixtures try to load data using `dvc.api` or local `data/data.csv` if present.

## Notes about cloud resources (where applicable)

- The original repo used GCS buckets and a PostgreSQL Cloud SQL backend for MLflow metadata and artifacts. If you will run in cloud, ensure the correct buckets/DB URIs and credentials are configured via environment variables or secrets.
- Example GCS bucket names referenced in the project (read-only here):
  - `gs://vertex-mlflow-artifacts-electric-wave-472614-d5`
  - `gs://mlops-week02-ga02-electric-wave-472614-d5`

## Troubleshooting

- DVC failures: double-check GCP authentication and remote configuration.
- MLflow server: confirm tracking URI and that the backend store (DB) is reachable and credentials are correct.
- Tests fail: ensure that a model is registered and DVC data is available locally.

## What I changed

- This file was refreshed for Week 6 graded assignment. It consolidates the earlier Week 5 contents and adds concise step-by-step notes for local development on Windows, Docker usage, and test expectations.

---

## Deployment & CI files (what they are and how to use them)

This repository includes three kinds of deployment-related artifacts: the `Dockerfile`, GitHub Actions CI/CD workflow(s) under `.github/workflows/`, and the Kubernetes manifests under `gke-deploy/`. Below is a concise explanation of what each file contains, what it does, and how you can use or improve them locally.

### `Dockerfile`
- Purpose: builds a container image for the FastAPI service in `app/` so the model can be served in Kubernetes or Docker.
- Key contents (what to expect):
  - Base image (Python 3.10 slim) and `WORKDIR` set to `/app`.
  - Installs dependencies listed in `req-app.txt` (keep this file minimal for the API image).
  - Copies the `app/` folder into the image so the FastAPI app and related modules are available at runtime.
  - Exposes port 8000 and sets the default command to run `uvicorn main:app --host 0.0.0.0 --port 8000`.
- How to build locally (PowerShell):

```powershell
docker build -t iris-api -f Dockerfile .
```

- How to run locally (PowerShell):

```powershell
docker run -p 8000:8000 --env MLFLOW_TRACKING_SERVER="http://<your-tracking-server>" iris-api
```

### GitHub Actions workflow(s) (`.github/workflows/*.yml`)
- Purpose: runs CI checks (install deps, run tests, pull DVC data, evaluate model, generate reports) and upload artifacts or comment PRs with results.
- Key contents to look for:
  - `actions/checkout` + `setup-python` for runner setup
  - `pip install` steps for dependencies (ensure `req.txt` contains required libs)
  - Authentication step (e.g., `google-github-actions/auth`) using a secret like `GCP_SA_KEY` for GCS/DVC access
  - DVC commands (`dvc pull`) to fetch the dataset during CI
  - `pytest` invocation for `tests/test_evaluation_1.py` which produces `cml_report.md` and other artifacts
  - Artifact upload step to GCS (using `gsutil` or `gcloud storage`) and optional CML comment creation
- How to run parts locally (adapting CI steps):

```powershell
# Create venv, install deps
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r req.txt

# Pull DVC data (requires dvc & configured remote)
dvc pull data/data.csv.dvc

# Run tests
pytest -q
```

### Kubernetes manifests (`gke-deploy/deployment.yaml` and `gke-deploy/service.yaml`)
- Purpose: deploy the built `iris-api` container into a Kubernetes cluster (GKE in this project) and expose it via a LoadBalancer service.
- Files and what they do:
  - `deployment.yaml`: declares a `Deployment` with a pod template that runs the container image, exposes container port (8000), and typically contains environment variables (e.g., `MLFLOW_TRACKING_SERVER`) so the app can contact MLflow. It may include resource requests/limits and liveness/readiness probes.
  - `service.yaml`: creates a `Service` of type `LoadBalancer` to expose the pods on a public IP and forward traffic to container port 8000.
- How to apply to a Kubernetes cluster (after pushing image to a registry and setting kubeconfig for the cluster):

```powershell
# Apply manifests
kubectl apply -f gke-deploy/deployment.yaml
kubectl apply -f gke-deploy/service.yaml

# Check rollout and service
kubectl rollout status deployment/iris-api-deployment
kubectl get svc iris-api-service -w
```

### Security & best-practice notes
- Do not hard-code sensitive values (DB URIs, MLflow server URIs, service account keys) in manifests—use Kubernetes `Secrets` or `ConfigMap` and reference them via `env.valueFrom.secretKeyRef` or `env.valueFrom.configMapKeyRef`.
- In CI, store sensitive credentials as GitHub Secrets (for example `GCP_SA_KEY` and `GCP_PROJECT_ID`) and only expose them to runs that need them.
- Prefer using tagged image names (e.g., `.../iris-api:v1.0.0`) and update `deployment.yaml` with the exact image tag used by CI to avoid pod drift.

## Key features

- ✅ Experiment tracking with MLflow (parameters, metrics, artifacts)
- ✅ Model registry for versioned model management
- ✅ Data versioning using DVC for reproducible datasets
- ✅ Automated evaluation and data validation (Pandera + pytest)
- ✅ CI/CD integration (GitHub Actions) to run tests, generate reports, and upload artifacts
- ✅ Containerized serving with a Dockerfile and Kubernetes manifests for GKE
- ✅ Configurable for cloud artifact stores (GCS) and managed metadata stores (Cloud SQL)

## Contact & support

If you need help or want to report an issue with this assignment:

- Open an issue in this repository (preferred)
- Contact the course instructor or teaching assistant as per course guidelines

Student: Ripunjay Kumar (Roll No: 21f3002511) — Week 6 graded assignment
