import os
import json
import pytest
import dvc.api
import pandera as pa
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import storage

# ================== CONFIG ==================
DATA_PATH = "../data/data.csv"
REPORT_PATH = "cml_report_poisoning.md"
CONF_MATRIX_PATH = "confusion_matrix_validation.png"
METRICS_PATH = "metrics_poisoning.json"
GCS_BUCKET = "gs://vertex-mlflow-artifacts-electric-wave-472614-d5"

# Poison detection thresholds
ACCURACY_THRESHOLD = 0.85
F1_THRESHOLD = 0.85

# ----- Pandera Schema (Enhanced for Poisoning Detection) -----
iris_schema = pa.DataFrameSchema({
    "sepal_length": pa.Column(float, pa.Check.in_range(4.0, 8.0)),
    "sepal_width": pa.Column(float, pa.Check.in_range(2.0, 5.0)),
    "petal_length": pa.Column(float, pa.Check.in_range(1.0, 7.0)),
    "petal_width": pa.Column(float, pa.Check.in_range(0.0, 3.0)),
    "species": pa.Column(str, pa.Check.isin(['setosa', 'versicolor', 'virginica']))
})

# ----- DVC Data Fixture -----
@pytest.fixture(scope="module")
def dvc_data():
    """Fetch data tracked by DVC."""
    with dvc.api.open(DATA_PATH, mode="r", encoding="utf-8") as f:
        df = pd.read_csv(f)
    return df

# ----- Data Validation Test with Anomaly Detection -----
def test_data_validation(dvc_data):
    """Validate dataset structure and check for anomalies."""
    print("\n🔍 Validating data schema...")
    
    validation_results = {
        "schema_valid": False,
        "anomalies_detected": [],
        "poison_indicators": []
    }
    
    try:
        iris_schema.validate(dvc_data, lazy=True)
        validation_results["schema_valid"] = True
        print("✅ Schema validation passed!")
    except pa.errors.SchemaErrors as err:
        validation_results["schema_valid"] = False
        validation_results["anomalies_detected"].append(str(err))
        print(f"⚠️  Schema validation failed:\n{err}")
    
    # Statistical anomaly detection
    print("\n🔬 Checking for statistical anomalies...")
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    for feature in features:
        feature_data = dvc_data[feature]
        
        # Z-score based outlier detection
        z_scores = np.abs((feature_data - feature_data.mean()) / feature_data.std())
        outliers = np.sum(z_scores > 3)
        outlier_pct = (outliers / len(feature_data)) * 100
        
        print(f"  {feature}: {outliers} outliers ({outlier_pct:.2f}%)")
        
        if outlier_pct > 5:  # More than 5% outliers indicates poisoning
            validation_results["poison_indicators"].append({
                "feature": feature,
                "outlier_percentage": outlier_pct,
                "severity": "HIGH" if outlier_pct > 15 else "MEDIUM"
            })
    
    # Label distribution check
    print("\n📊 Checking label distribution...")
    label_counts = dvc_data['species'].value_counts()
    total = len(dvc_data)
    
    for species, count in label_counts.items():
        percentage = (count / total) * 100
        print(f"  {species}: {count} ({percentage:.1f}%)")
        
        # IRIS should be balanced (~33% each)
        if percentage < 25 or percentage > 40:
            validation_results["poison_indicators"].append({
                "type": "label_imbalance",
                "class": species,
                "percentage": percentage,
                "severity": "MEDIUM"
            })
    
    # Generate validation report
    with open("data_validation_report.json", "w") as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n{'='*60}")
    if validation_results["poison_indicators"]:
        print("⚠️  POTENTIAL DATA POISONING DETECTED!")
        for indicator in validation_results["poison_indicators"]:
            print(f"  - {indicator}")
    else:
        print("✅ No poisoning indicators detected")
    print(f"{'='*60}")
    
    # Fail test if schema validation fails
    if not validation_results["schema_valid"]:
        pytest.fail("❌ Data validation failed due to schema errors")

# ----- Load Latest MLflow Model from GCS -----
def get_latest_model_from_gcs(bucket_uri):
    """Fetch the latest MLflow model artifact folder from GCS."""
    print(f"\n🔍 Fetching latest MLflow model from {bucket_uri} ...")
    client = storage.Client()

    bucket_name = bucket_uri.replace("gs://", "").split("/")[0]
    prefix = "/".join(bucket_uri.replace("gs://", "").split("/")[1:])
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    # Identify all folders containing MLmodel
    model_folders = [b.name.rsplit("/", 1)[0] for b in blobs if b.name.endswith("MLmodel")]
    if not model_folders:
        raise FileNotFoundError(f"No MLflow model found under {bucket_uri}")

    # Pick the latest by timestamp
    latest_model_blob = max(
        [b for b in blobs if b.name.endswith("MLmodel")],
        key=lambda x: x.updated
    )
    latest_model_folder = latest_model_blob.name.rsplit("/", 1)[0]

    model_uri = f"gs://{bucket_name}/{latest_model_folder}"
    print(f"✅ Latest model folder detected: {model_uri}")

    # Load model using MLflow
    model = mlflow.sklearn.load_model(model_uri)
    return model, model_uri

# ----- Model Evaluation Test with Poisoning Analysis -----
def test_model_evaluation(dvc_data):
    """Evaluate the latest MLflow model and check for poisoning effects."""
    print("\n🚀 Evaluating latest MLflow model...")
    model, model_uri = get_latest_model_from_gcs(GCS_BUCKET)

    # === UPDATE: Parse Run ID from Model URI ===
    run_id = "unknown"
    try:
        # Assumes URI structure: gs://.../<experiment_id>/<run_id>/artifacts/model
        path_parts = model_uri.split("/")
        if len(path_parts) > 2 and path_parts[-2] == 'artifacts':
            run_id = path_parts[-3]
            print(f"✅ Testing MLflow Run ID: {run_id}")
    except Exception as e:
        print(f"⚠️  Could not parse Run ID from URI: {e}")
    # ==========================================

    X = dvc_data.drop("species", axis=1).values
    y_true = dvc_data["species"].values

    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"🎯 Evaluation Results:")
    print(f"  - Run ID: {run_id}") # <-- ADDED RUN ID
    print(f"  - Accuracy: {acc:.4f}")
    print(f"  - F1 Score: {f1:.4f}")
    print(f"{'='*60}")

    # Per-class analysis
    print("\n📊 Per-Class Performance:")
    classes = np.unique(y_true)
    class_metrics = {}
    
    for i, cls in enumerate(classes):
        cls_mask = (y_true == cls)
        cls_acc = accuracy_score(y_true[cls_mask], y_pred[cls_mask])
        class_metrics[cls] = cls_acc
        print(f"  {cls}: {cls_acc:.4f}")
    
    # Detect performance degradation
    degradation_detected = False
    if acc < ACCURACY_THRESHOLD:
        degradation_detected = True
        print(f"\n⚠️  WARNING: Accuracy below threshold ({ACCURACY_THRESHOLD})")
    if f1 < F1_THRESHOLD:
        degradation_detected = True
        print(f"⚠️  WARNING: F1 Score below threshold ({F1_THRESHOLD})")
    
    # Check for class-specific issues
    for cls, cls_acc in class_metrics.items():
        if cls_acc < 0.7:
            print(f"⚠️  WARNING: Poor performance on class '{cls}' ({cls_acc:.4f})")
            degradation_detected = True

    # ---- Confusion Matrix Plot ----
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    # <-- UPDATED TITLE WITH RUN ID -->
    plt.title(f"Confusion Matrix — Iris Classifier\nRun ID: {run_id}\nAccuracy: {acc:.4f}, F1: {f1:.4f}", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_PATH, dpi=150)
    print(f"\n📊 Saved confusion matrix to {CONF_MATRIX_PATH}")

    # ---- CML Markdown Report ----
    with open(REPORT_PATH, "w") as f:
        f.write("# Model Evaluation Report — Data Poisoning Study\n\n")
        f.write(f"**Run ID:** `{run_id}`\n") # <-- ADDED RUN ID
        f.write(f"**Model URI:** `{model_uri}`\n\n")
        f.write("## Overall Metrics\n\n")
        f.write(f"| Metric | Value | Status |\n")
        f.write(f"|--------|-------|--------|\n")
        f.write(f"| Accuracy | `{acc:.4f}` | {'✅ PASS' if acc >= ACCURACY_THRESHOLD else '❌ FAIL'} |\n")
        f.write(f"| F1 Score | `{f1:.4f}` | {'✅ PASS' if f1 >= F1_THRESHOLD else '❌ FAIL'} |\n\n")
        
        f.write("## Per-Class Performance\n\n")
        f.write("| Class | Accuracy |\n")
        f.write("|-------|----------|\n")
        for cls, cls_acc in class_metrics.items():
            f.write(f"| {cls} | `{cls_acc:.4f}` |\n")
        
        f.write("\n## Confusion Matrix\n\n")
        f.write(f"![Confusion Matrix]({CONF_MATRIX_PATH})\n\n")
        
        if degradation_detected:
            f.write("## ⚠️  Alerts\n\n")
            f.write("**Potential data poisoning or model degradation detected!**\n\n")
            f.write("- Model performance is below acceptable thresholds\n")
            f.write("- Review training data for anomalies\n")
            f.write("- Check data validation reports\n")
            f.write("- Consider retraining with clean data\n")

    # ---- Metrics JSON for CI/CD ----
    metrics = {
        "run_id": run_id, # <-- ADDED RUN ID
        "accuracy": acc,
        "f1_score": f1,
        "class_metrics": class_metrics,
        "degradation_detected": degradation_detected,
        "passed_thresholds": acc >= ACCURACY_THRESHOLD and f1 >= F1_THRESHOLD
    }
    
    with open(METRICS_PATH, "w") as jf:
        json.dump(metrics, jf, indent=2)

    print(f"\n✅ Evaluation report saved to {REPORT_PATH}")
    print(f"✅ Metrics saved to {METRICS_PATH}")

    # ---- CI/CD Assertions ----
    if degradation_detected:
        print("\n❌ TEST FAILED: Model performance degradation detected!")
        pytest.fail(
            f"Model performance below thresholds. "
            f"Accuracy: {acc:.4f} (threshold: {ACCURACY_THRESHOLD}), "
            f"F1: {f1:.4f} (threshold: {F1_THRESHOLD})"
        )
    else:
        print("\n✅ TEST PASSED: Model meets performance requirements")

# ----- Robustness Test Against Known Poisoning -----
def test_model_robustness(dvc_data):
    """Test model robustness against synthetic poisoning attacks."""
    print("\n🛡️  Testing model robustness against poisoning...")
    model, _ = get_latest_model_from_gcs(GCS_BUCKET)
    
    X_clean = dvc_data.drop("species", axis=1).values
    y_clean = dvc_data["species"].values
    
    # Test 1: Feature noise injection
    X_noisy = X_clean + np.random.normal(0, 0.5, X_clean.shape)
    y_pred_noisy = model.predict(X_noisy)
    acc_noisy = accuracy_score(y_clean, y_pred_noisy)
    
    print(f"  Accuracy with noise: {acc_noisy:.4f}")
    
    # Test 2: Feature scaling attack
    X_scaled = X_clean * np.random.uniform(0.8, 1.2, X_clean.shape)
    y_pred_scaled = model.predict(X_scaled)
    acc_scaled = accuracy_score(y_clean, y_pred_scaled)
    
    print(f"  Accuracy with scaling: {acc_scaled:.4f}")
    
    robustness_score = min(acc_noisy, acc_scaled)
    print(f"\n  Overall robustness score: {robustness_score:.4f}")
    
    if robustness_score < 0.7:
        print("  ⚠️  Model shows low robustness to input perturbations")
    else:
        print("  ✅ Model shows good robustness")
    
    # Save robustness results
    robustness_metrics = {
        "accuracy_with_noise": acc_noisy,
        "accuracy_with_scaling": acc_scaled,
        "robustness_score": robustness_score
    }
    
    with open("robustness_metrics.json", "w") as f:
        json.dump(robustness_metrics, f, indent=2)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])