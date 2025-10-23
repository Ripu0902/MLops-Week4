# MLOps Week 5 Graded Assignment

**Name:** Ripunjay Kumar  
**Roll No:** 21f3002511  
**Term:** Sept 2025  
**Course:** MLOps (BSDA5014)  
**Lab Done by:** Ripunjay Kumar

## MLOps Pipeline with MLflow Experiment Tracking & Model Registry

This repository implements a complete MLOps pipeline for Iris classification with integrated experiment tracking, model registry, automated testing, and CI/CD deployment. The pipeline combines MLflow for experiment management, DVC for data versioning, PostgreSQL Cloud SQL for MLflow backend, and automated testing with CML reporting.

### Architecture Overview

- **MLflow Tracking Server**: PostgreSQL Cloud SQL backend for experiment metadata
- **Artifact Storage**: Two GCS buckets - one for DVC data versioning, another for MLflow model artifacts
- **Model Training**: Decision Tree classifier with hyperparameter experiments logged to MLflow
- **Model Registry**: Automatic registration of best-performing models
- **CI/CD Pipeline**: Automated testing, evaluation, and reporting with GitHub Actions
- **Data Validation**: Schema validation using Pandera
- **Model Evaluation**: Automated evaluation of latest models from GCS artifacts


## Project Structure

```
├── .dvc/                          # DVC configuration
│   └── config                     # DVC remote configuration (GCS bucket)
├── .github/workflows/             # CI/CD pipeline
│   └── ci-cd.yml                  # GitHub Actions workflow
├── data/                          # Dataset directory
│   ├── data.csv                   # Iris dataset
│   └── data.csv.dvc              # DVC tracking file
├── src/                           # Source code
│   └── evaluate_latest_model_mlflow.py  # Model evaluation script
├── tests/                         # Test suite
│   └── test_evaluation_1.py       # Data validation & model evaluation tests
├── model/                         # Local model artifacts
├── metrics/                       # Evaluation metrics
├── train_iris_dt_mlflow.py        # MLflow training script
├── req.txt                        # Python dependencies
└── README.md                      # This file
```

## Key Components

### 1. MLflow Integration
- **Training Script**: [`train_iris_dt_mlflow.py`](train_iris_dt_mlflow.py)
  - Hyperparameter experiments with Decision Tree classifier
  - Automatic logging of parameters, metrics, and model artifacts
  - Model signature inference and input examples
  - Automatic registration of best-performing models to MLflow Model Registry

### 2. Data & Model Management
- **DVC Configuration**: [`.dvc/config`](.dvc/config) - GCS bucket for data versioning
- **MLflow Artifacts**: Separate GCS bucket (`gs://vertex-mlflow-artifacts-electric-wave-472614-d5`) for model storage
- **PostgreSQL Cloud SQL**: Backend database for MLflow tracking server metadata

### 3. Testing & Validation
- **Test Suite**: [`tests/test_evaluation_1.py`](tests/test_evaluation_1.py)
  - Data validation using Pandera schema
  - Automated model evaluation from latest GCS artifacts
  - Confusion matrix generation and CML reporting
- **Evaluation Script**: [`src/evaluate_latest_model_mlflow.py`](src/evaluate_latest_model_mlflow.py)
  - Standalone model evaluation using MLflow Model Registry

### 4. CI/CD Pipeline
- **Workflow**: [`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml)
  - Triggers on pushes to `dev`, `main`, `week5` branches and PRs
  - Automated data pulling from DVC
  - Model evaluation from MLflow artifacts in GCS
  - CML report generation and GitHub PR comments
  - Artifact upload to GCS bucket

## Workflow Details

### Training Process
1. **Data Retrieval**: DVC pulls the latest Iris dataset from GCS bucket
2. **Experiment Tracking**: MLflow logs multiple hyperparameter configurations
3. **Model Registration**: Best-performing model automatically registered to MLflow Model Registry
4. **Artifact Storage**: Model artifacts stored in dedicated GCS bucket

### Testing & Evaluation
- **`test_data_validation`**: Validates dataset structure using Pandera schema
  - Checks data types, value ranges, and categorical constraints
  - Ensures data quality before model evaluation

- **`test_model_evaluation`**: Evaluates latest MLflow model from GCS artifacts
  - Automatically detects and loads the most recent model from GCS bucket
  - Computes accuracy and F1-score metrics
  - Generates confusion matrix visualization
  - Creates CML markdown report for GitHub PR comments
  - Asserts model performance thresholds (accuracy > 0.8, F1 > 0.8)

### CI/CD Pipeline Flow
1. **Trigger**: Push to `dev`/`main`/`week5` branches or PR creation
2. **Environment Setup**: Python 3.10, dependency installation
3. **Authentication**: GCP service account authentication via `GCP_SA_KEY` secret
4. **Data Pipeline**: DVC configuration and data pulling from GCS
5. **Testing**: Automated data validation and model evaluation
6. **Reporting**: CML report generation and GitHub PR commenting
7. **Artifact Management**: Upload evaluation reports to GCS bucket with timestamps

## Local Development Setup

### Prerequisites
- Python 3.10+
- Google Cloud SDK with authentication
- Access to GCS buckets and PostgreSQL Cloud SQL instance

### Installation & Setup
1. **Clone repository and create virtual environment**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r req.txt pytest pandera scikit-learn pandas matplotlib mlflow seaborn google-cloud-storage
   ```

3. **Configure environment variables**:
   ```bash
   # Create .env file with MLflow tracking server URI
   echo "MLFLOW_TRACKING_SERVER=<your-mlflow-server-uri>" > .env
   ```

4. **Setup DVC and pull data**:
   ```bash
   dvc remote default gcsremote
   dvc pull
   ```

### Running Components

#### Train Model with MLflow
```bash
python train_iris_dt_mlflow.py
```

#### Run Tests Locally
```bash
pytest tests/test_evaluation_1.py -v
```

#### Evaluate Latest Model
```bash
python src/evaluate_latest_model_mlflow.py
```

## Infrastructure Configuration

### GCS Buckets
- **DVC Data Storage**: `gs://mlops-week02-ga02-electric-wave-472614-d5`
  - Stores versioned datasets and model artifacts for DVC tracking
- **MLflow Artifacts**: `gs://vertex-mlflow-artifacts-electric-wave-472614-d5`
  - Stores MLflow model artifacts and experiment data

### MLflow Setup
- **Backend Store**: PostgreSQL Cloud SQL instance for metadata storage
- **Artifact Store**: GCS bucket for model artifacts and experiment files
- **Model Registry**: Centralized model versioning and lifecycle management

### GitHub Secrets Required
- `GCP_SA_KEY`: Google Cloud service account JSON for authentication
- `GITHUB_TOKEN`: Automatically provided for CML report commenting

## Troubleshooting

### Common Issues
- **DVC Pull Failures**: Ensure GCP authentication and bucket access permissions
- **Test Failures**: Check model availability in GCS bucket and data schema compliance
- **CI/CD Issues**: Review GitHub Actions logs and GCP service account permissions

### Model Evaluation Criteria
- **Accuracy Threshold**: > 0.8 (80%)
- **F1-Score Threshold**: > 0.8 (80%)
- **Data Validation**: Strict Pandera schema compliance required

## Key Features

- ✅ **Experiment Tracking**: Complete MLflow integration with PostgreSQL backend
- ✅ **Model Registry**: Automatic registration of best-performing models
- ✅ **Data Versioning**: DVC integration with GCS for reproducible datasets
- ✅ **Automated Testing**: Comprehensive data validation and model evaluation
- ✅ **CI/CD Pipeline**: GitHub Actions with automated reporting
- ✅ **Cloud Integration**: GCS buckets for scalable artifact storage
- ✅ **Performance Monitoring**: Automated model performance assertions
- ✅ **Visualization**: Confusion matrix generation and CML reporting

## Contact & Support

**Student**: Ripunjay Kumar (21f3002511)  
**Course**: MLOps (BSDA5014) - Sept 2025  
**Assignment**: Week 5 Graded Assignment

For issues or questions, check the GitHub Actions logs and ensure all cloud resources are properly configured and accessible.