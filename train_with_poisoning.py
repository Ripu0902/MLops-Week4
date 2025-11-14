#!/usr/bin/env python3
"""
Train DecisionTreeClassifier on IRIS dataset with data poisoning at various levels.
Demonstrates impact of data quality on model performance using MLflow tracking.
"""

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
import subprocess
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

load_dotenv()

# ======================
# CONFIGURATION
# ======================
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_SERVER")
EXPERIMENT_NAME = "iris_data_poisoning"
DATA_CSV = "data/data.csv"
DATA_DVC_FILE = "data/data.csv.dvc"

# Poisoning levels to test
POISON_LEVELS = [0.0, 0.05, 0.10, 0.25, 0.50]

# ======================
# DATA POISONING FUNCTIONS
# ======================
def poison_features(X, poison_rate, method='random_noise'):
    """
    Poison features by injecting random noise or outliers.
    
    Args:
        X: Feature matrix (numpy array)
        poison_rate: Percentage of samples to poison (0.0 to 1.0)
        method: 'random_noise' or 'outliers'
    
    Returns:
        Poisoned feature matrix
    """
    X_poisoned = X.copy()
    n_samples = X.shape[0]
    n_poison = int(n_samples * poison_rate)
    
    if n_poison == 0:
        return X_poisoned
    
    # Randomly select indices to poison
    poison_indices = np.random.choice(n_samples, size=n_poison, replace=False)
    
    if method == 'random_noise':
        # Replace with random values from uniform distribution
        for idx in poison_indices:
            X_poisoned[idx] = np.random.uniform(
                low=X.min(axis=0) * 0.5,
                high=X.max(axis=0) * 1.5,
                size=X.shape[1]
            )
    elif method == 'outliers':
        # Inject extreme outliers
        for idx in poison_indices:
            X_poisoned[idx] = np.random.uniform(
                low=X.max(axis=0) * 2,
                high=X.max(axis=0) * 4,
                size=X.shape[1]
            )
    
    return X_poisoned


def poison_labels(y, poison_rate):
    """
    Poison labels by randomly flipping them to incorrect classes.
    
    Args:
        y: Label array
        poison_rate: Percentage of labels to flip (0.0 to 1.0)
    
    Returns:
        Poisoned label array
    """
    y_poisoned = y.copy()
    n_samples = len(y)
    n_poison = int(n_samples * poison_rate)
    
    if n_poison == 0:
        return y_poisoned
    
    # Randomly select indices to poison
    poison_indices = np.random.choice(n_samples, size=n_poison, replace=False)
    
    # Get unique classes
    unique_classes = np.unique(y)
    
    # Flip labels to random wrong class
    for idx in poison_indices:
        current_label = y_poisoned[idx]
        # Choose different label
        possible_labels = unique_classes[unique_classes != current_label]
        y_poisoned[idx] = np.random.choice(possible_labels)
    
    return y_poisoned


def create_confusion_matrix_plot(y_true, y_pred, poison_level, plot_type='train'):
    """Create and save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['setosa', 'versicolor', 'virginica'],
                yticklabels=['setosa', 'versicolor', 'virginica'])
    plt.title(f'Confusion Matrix - {plot_type.upper()} ({poison_level*100:.0f}% Poisoning)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename = f"confusion_matrix_{plot_type}_{poison_level:.2f}.png"
    plt.savefig(filename)
    plt.close()
    return filename


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
# LOAD CLEAN DATA
# ======================
df = pd.read_csv(DATA_CSV)
X_clean = df.drop("species", axis=1).values
y_clean = df["species"].values

print(f"\n📊 Dataset loaded: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
print(f"Classes: {np.unique(y_clean)}")

# ======================
# LOG DATA VERSION INFO
# ======================
git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
dvc_status = subprocess.check_output(["dvc", "status", DATA_DVC_FILE]).decode().strip()

# ======================
# EXPERIMENT: FEATURE POISONING
# ======================
print("\n" + "="*70)
print("EXPERIMENT 1: FEATURE POISONING (Random Noise)")
print("="*70)

for poison_level in POISON_LEVELS:
    print(f"\n🧪 Testing with {poison_level*100:.0f}% feature poisoning...")
    
    # Apply poisoning to features
    X_poisoned = poison_features(X_clean, poison_level, method='random_noise')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_poisoned, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    
    # Also split clean data for validation comparison
    X_train_clean, X_test_clean, _, _ = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    
    with mlflow.start_run(run_name=f"feature_poison_{poison_level*100:.0f}pct"):
        # Train model on poisoned data
        clf = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate on poisoned test set
        y_pred_test = clf.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test, average='macro')
        precision_test = precision_score(y_test, y_pred_test, average='macro')
        recall_test = recall_score(y_test, y_pred_test, average='macro')
        
        # Evaluate on clean test set (robustness check)
        y_pred_clean = clf.predict(X_test_clean)
        acc_clean = accuracy_score(y_test, y_pred_clean)
        f1_clean = f1_score(y_test, y_pred_clean, average='macro')
        
        # Training accuracy
        y_pred_train = clf.predict(X_train)
        acc_train = accuracy_score(y_train, y_pred_train)
        
        # Log parameters
        mlflow.log_param("poison_type", "feature_noise")
        mlflow.log_param("poison_level", poison_level)
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("criterion", "gini")
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("dvc_status", dvc_status)
        
        # Log metrics
        mlflow.log_metric("train_accuracy", acc_train)
        mlflow.log_metric("test_accuracy_poisoned", acc_test)
        mlflow.log_metric("test_accuracy_clean", acc_clean)
        mlflow.log_metric("test_f1_poisoned", f1_test)
        mlflow.log_metric("test_f1_clean", f1_clean)
        mlflow.log_metric("test_precision", precision_test)
        mlflow.log_metric("test_recall", recall_test)
        mlflow.log_metric("accuracy_degradation", acc_clean - acc_test)
        
        # Create and log confusion matrices
        cm_train = create_confusion_matrix_plot(y_train, y_pred_train, poison_level, 'train')
        cm_test = create_confusion_matrix_plot(y_test, y_pred_test, poison_level, 'test')
        mlflow.log_artifact(cm_train)
        mlflow.log_artifact(cm_test)
        
        # Log model
        signature = infer_signature(X_train, clf.predict(X_train))
        mlflow.sklearn.log_model(clf, "model", signature=signature)
        
        print(f"  ✅ Train Acc: {acc_train:.4f}")
        print(f"  📉 Test Acc (Poisoned): {acc_test:.4f}")
        print(f"  📈 Test Acc (Clean): {acc_clean:.4f}")
        print(f"  ⚠️  Degradation: {(acc_clean - acc_test):.4f}")

# ======================
# EXPERIMENT: LABEL POISONING
# ======================
print("\n" + "="*70)
print("EXPERIMENT 2: LABEL POISONING (Random Label Flips)")
print("="*70)

for poison_level in POISON_LEVELS:
    print(f"\n🧪 Testing with {poison_level*100:.0f}% label poisoning...")
    
    # Apply poisoning to labels
    y_poisoned = poison_labels(y_clean, poison_level)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_poisoned, test_size=0.2, random_state=42, stratify=y_clean
    )
    
    # Clean labels for comparison
    _, _, y_train_clean, y_test_clean = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    
    with mlflow.start_run(run_name=f"label_poison_{poison_level*100:.0f}pct"):
        # Train model on data with poisoned labels
        clf = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate on poisoned labels
        y_pred_test = clf.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test, average='macro')
        
        # Evaluate against clean labels (true performance)
        acc_test_true = accuracy_score(y_test_clean, y_pred_test)
        f1_test_true = f1_score(y_test_clean, y_pred_test, average='macro')
        
        # Training metrics
        y_pred_train = clf.predict(X_train)
        acc_train_poisoned = accuracy_score(y_train, y_pred_train)
        acc_train_true = accuracy_score(y_train_clean, y_pred_train)
        
        # Log parameters
        mlflow.log_param("poison_type", "label_flip")
        mlflow.log_param("poison_level", poison_level)
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("criterion", "gini")
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("dvc_status", dvc_status)
        
        # Log metrics
        mlflow.log_metric("train_accuracy_poisoned", acc_train_poisoned)
        mlflow.log_metric("train_accuracy_true", acc_train_true)
        mlflow.log_metric("test_accuracy_poisoned", acc_test)
        mlflow.log_metric("test_accuracy_true", acc_test_true)
        mlflow.log_metric("test_f1_poisoned", f1_test)
        mlflow.log_metric("test_f1_true", f1_test_true)
        mlflow.log_metric("label_confusion_rate", poison_level)
        
        # Create confusion matrices
        cm_train_poi = create_confusion_matrix_plot(y_train, y_pred_train, poison_level, 'train_poisoned')
        cm_train_true = create_confusion_matrix_plot(y_train_clean, y_pred_train, poison_level, 'train_true')
        mlflow.log_artifact(cm_train_poi)
        mlflow.log_artifact(cm_train_true)
        
        # Log model
        signature = infer_signature(X_train, clf.predict(X_train))
        mlflow.sklearn.log_model(clf, "model", signature=signature)
        
        print(f"  ✅ Train Acc (vs poisoned): {acc_train_poisoned:.4f}")
        print(f"  ⚠️  Train Acc (vs true): {acc_train_true:.4f}")
        print(f"  📉 Test Acc (vs poisoned): {acc_test:.4f}")
        print(f"  📈 Test Acc (vs true): {acc_test_true:.4f}")

# ======================
# GENERATE SUMMARY REPORT
# ======================
print("\n" + "="*70)
print("📊 GENERATING SUMMARY REPORT")
print("="*70)

# Fetch all runs
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
all_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"]
)

# Create summary dataframe
summary_data = []
for run in all_runs:
    metrics = run.data.metrics
    params = run.data.params
    summary_data.append({
        'run_name': run.info.run_name,
        'poison_type': params.get('poison_type', 'N/A'),
        'poison_level': float(params.get('poison_level', 0)),
        'train_accuracy': metrics.get('train_accuracy', metrics.get('train_accuracy_true', 0)),
        'test_accuracy': metrics.get('test_accuracy_poisoned', metrics.get('test_accuracy_true', 0)),
        'f1_score': metrics.get('test_f1_poisoned', metrics.get('test_f1_true', 0))
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values(['poison_type', 'poison_level'])

print("\n📋 Summary Table:")
print(summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('poisoning_summary.csv', index=False)
print(f"\n✅ Summary saved to 'poisoning_summary.csv'")
print(f"🔗 View results in MLflow UI: {TRACKING_URI}")
print(f"📁 Experiment: {EXPERIMENT_NAME}")