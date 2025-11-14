#!/usr/bin/env python3
"""
Analyze and visualize the impact of data poisoning from MLflow experiments.
Generates comprehensive reports and visualizations.
"""

import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

load_dotenv()

# ======================
# CONFIGURATION
# ======================
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_SERVER")
EXPERIMENT_NAME = "iris_data_poisoning"

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# ======================
# CONNECT TO MLflow
# ======================
mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.tracking.MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    print(f"❌ Experiment '{EXPERIMENT_NAME}' not found!")
    print("Run train_with_poisoning.py first.")
    exit(1)

print(f"📊 Analyzing experiment: {EXPERIMENT_NAME}")
print(f"🔗 MLflow URI: {TRACKING_URI}")

# ======================
# FETCH ALL RUNS
# ======================
all_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"]
)

print(f"\n✅ Found {len(all_runs)} experiment runs")

# ======================
# EXTRACT DATA
# ======================
data = []
for run in all_runs:
    metrics = run.data.metrics
    params = run.data.params
    
    poison_type = params.get('poison_type', 'N/A')
    poison_level = float(params.get('poison_level', 0))
    
    data.append({
        'run_id': run.info.run_id,
        'run_name': run.info.run_name,
        'poison_type': poison_type,
        'poison_level': poison_level,
        'poison_pct': poison_level * 100,
        'train_accuracy': metrics.get('train_accuracy', metrics.get('train_accuracy_true', np.nan)),
        'test_accuracy_poisoned': metrics.get('test_accuracy_poisoned', np.nan),
        'test_accuracy_clean': metrics.get('test_accuracy_clean', np.nan),
        'test_accuracy_true': metrics.get('test_accuracy_true', np.nan),
        'f1_poisoned': metrics.get('test_f1_poisoned', np.nan),
        'f1_clean': metrics.get('test_f1_clean', np.nan),
        'f1_true': metrics.get('test_f1_true', np.nan),
        'accuracy_degradation': metrics.get('accuracy_degradation', np.nan)
    })

df = pd.DataFrame(data)

# Separate by poison type
df_feature = df[df['poison_type'] == 'feature_noise'].sort_values('poison_level')
df_label = df[df['poison_type'] == 'label_flip'].sort_values('poison_level')

print(f"\n📈 Feature Poisoning runs: {len(df_feature)}")
print(f"📈 Label Poisoning runs: {len(df_label)}")

# ======================
# CREATE COMPREHENSIVE VISUALIZATION
# ======================
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('Data Poisoning Impact Analysis — IRIS Dataset', 
             fontsize=20, fontweight='bold', y=0.98)

# -----------------------------
# PLOT 1: Feature Poisoning Impact
# -----------------------------
ax1 = fig.add_subplot(gs[0, :2])
if not df_feature.empty:
    ax1.plot(df_feature['poison_pct'], df_feature['train_accuracy'], 
             marker='o', linewidth=2, markersize=8, label='Train Accuracy', color='#2E86AB')
    ax1.plot(df_feature['poison_pct'], df_feature['test_accuracy_clean'], 
             marker='s', linewidth=2, markersize=8, label='Test Acc (Clean Data)', color='#06A77D')
    ax1.plot(df_feature['poison_pct'], df_feature['test_accuracy_poisoned'], 
             marker='^', linewidth=2, markersize=8, label='Test Acc (Poisoned Data)', color='#D62246')
    
    ax1.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Target Threshold (0.9)')
    ax1.set_xlabel('Poisoning Level (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Poisoning: Accuracy vs Poisoning Level', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.3, 1.05])

# -----------------------------
# PLOT 2: Label Poisoning Impact
# -----------------------------
ax2 = fig.add_subplot(gs[1, :2])
if not df_label.empty:
    ax2.plot(df_label['poison_pct'], df_label['train_accuracy'], 
             marker='o', linewidth=2, markersize=8, label='Train Acc (vs Poisoned Labels)', color='#2E86AB')
    ax2.plot(df_label['poison_pct'], df_label['test_accuracy_true'], 
             marker='s', linewidth=2, markersize=8, label='Test Acc (vs True Labels)', color='#06A77D')
    ax2.plot(df_label['poison_pct'], df_label['test_accuracy_poisoned'], 
             marker='^', linewidth=2, markersize=8, label='Test Acc (vs Poisoned Labels)', color='#D62246')
    
    ax2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Target Threshold (0.9)')
    ax2.set_xlabel('Poisoning Level (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Label Poisoning: Accuracy vs Poisoning Level', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.3, 1.05])

# -----------------------------
# PLOT 3: F1 Score Comparison
# -----------------------------
ax3 = fig.add_subplot(gs[0, 2])
if not df_feature.empty:
    ax3.plot(df_feature['poison_pct'], df_feature['f1_clean'], 
             marker='o', linewidth=2, markersize=6, label='Feature Poison', color='#06A77D')
if not df_label.empty:
    ax3.plot(df_label['poison_pct'], df_label['f1_true'], 
             marker='s', linewidth=2, markersize=6, label='Label Poison', color='#D62246')

ax3.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5)
ax3.set_xlabel('Poisoning Level (%)', fontsize=11, fontweight='bold')
ax3.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
ax3.set_title('F1 Score Degradation', fontsize=12, fontweight='bold')
ax3.legend(loc='best', frameon=True)
ax3.grid(True, alpha=0.3)

# -----------------------------
# PLOT 4: Accuracy Degradation (Feature)
# -----------------------------
ax4 = fig.add_subplot(gs[1, 2])
if not df_feature.empty and 'accuracy_degradation' in df_feature.columns:
    bars = ax4.bar(df_feature['poison_pct'], df_feature['accuracy_degradation'], 
                   color='#F77F00', edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Poisoning Level (%)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Accuracy Drop', fontsize=11, fontweight='bold')
    ax4.set_title('Performance Degradation', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

# -----------------------------
# PLOT 5: Summary Statistics Table
# -----------------------------
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')

# Create summary table
summary_text = []
summary_text.append("="*100)
summary_text.append("SUMMARY FINDINGS")
summary_text.append("="*100)

if not df_feature.empty:
    clean_acc_50 = df_feature[df_feature['poison_pct'] == 50]['test_accuracy_clean'].values
    clean_acc_0 = df_feature[df_feature['poison_pct'] == 0]['test_accuracy_clean'].values
    
    if len(clean_acc_50) > 0 and len(clean_acc_0) > 0:
        degradation = (clean_acc_0[0] - clean_acc_50[0]) * 100
        summary_text.append(f"\n🔍 FEATURE POISONING (Random Noise):")
        summary_text.append(f"   • At 0% poisoning: {clean_acc_0[0]:.4f} accuracy")
        summary_text.append(f"   • At 50% poisoning: {clean_acc_50[0]:.4f} accuracy")
        summary_text.append(f"   • Performance drop: {degradation:.2f}%")
        summary_text.append(f"   • Impact: {'SEVERE' if degradation > 20 else 'MODERATE' if degradation > 10 else 'MILD'}")

if not df_label.empty:
    true_acc_50 = df_label[df_label['poison_pct'] == 50]['test_accuracy_true'].values
    true_acc_0 = df_label[df_label['poison_pct'] == 0]['test_accuracy_true'].values
    
    if len(true_acc_50) > 0 and len(true_acc_0) > 0:
        degradation = (true_acc_0[0] - true_acc_50[0]) * 100
        summary_text.append(f"\n🏷️  LABEL POISONING (Random Flips):")
        summary_text.append(f"   • At 0% poisoning: {true_acc_0[0]:.4f} accuracy")
        summary_text.append(f"   • At 50% poisoning: {true_acc_50[0]:.4f} accuracy")
        summary_text.append(f"   • Performance drop: {degradation:.2f}%")
        summary_text.append(f"   • Impact: {'SEVERE' if degradation > 30 else 'MODERATE' if degradation > 15 else 'MILD'}")

summary_text.append("\n" + "="*100)
summary_text.append("KEY INSIGHTS:")
summary_text.append("="*100)
summary_text.append("1. Label poisoning has MORE severe impact than feature poisoning")
summary_text.append("2. Even 5-10% poisoning can significantly degrade model performance")
summary_text.append("3. Training accuracy may remain high even with poisoned data (misleading!)")
summary_text.append("4. Validation on clean data is CRITICAL to detect poisoning")
summary_text.append("5. Production models MUST include data quality checks and monitoring")

text_str = '\n'.join(summary_text)
ax5.text(0.05, 0.95, text_str, transform=ax5.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Save figure
output_file = 'poisoning_analysis_dashboard.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Dashboard saved to: {output_file}")

# ======================
# CREATE DETAILED COMPARISON TABLE
# ======================
print("\n" + "="*100)
print("DETAILED RESULTS TABLE")
print("="*100)

if not df_feature.empty:
    print("\n📊 FEATURE POISONING RESULTS:")
    print("-" * 100)
    feature_table = df_feature[['poison_pct', 'train_accuracy', 
                                 'test_accuracy_clean', 'test_accuracy_poisoned', 
                                 'accuracy_degradation']].copy()
    feature_table.columns = ['Poison%', 'Train Acc', 'Test Acc (Clean)', 
                             'Test Acc (Poisoned)', 'Degradation']
    print(feature_table.to_string(index=False))

if not df_label.empty:
    print("\n📊 LABEL POISONING RESULTS:")
    print("-" * 100)
    label_table = df_label[['poison_pct', 'train_accuracy', 
                            'test_accuracy_true', 'test_accuracy_poisoned']].copy()
    label_table.columns = ['Poison%', 'Train Acc (vs Poison)', 
                           'Test Acc (vs True)', 'Test Acc (vs Poison)']
    print(label_table.to_string(index=False))

# ======================
# RECOMMENDATIONS
# ======================
print("\n" + "="*100)
print("🛡️  RECOMMENDATIONS FOR PRODUCTION")
print("="*100)
print("""
1. DATA VALIDATION:
   ✓ Implement Pandera schemas with strict value ranges
   ✓ Use statistical tests (Z-scores, IQR) to detect outliers
   ✓ Monitor label distribution for unexpected shifts
   ✓ Track data provenance and lineage

2. MONITORING:
   ✓ Set up accuracy thresholds and alerts
   ✓ Compare validation vs training metrics
   ✓ Track per-class performance
   ✓ Monitor prediction confidence distributions

3. DEFENSE MECHANISMS:
   ✓ Use robust training methods (e.g., label smoothing)
   ✓ Apply outlier removal during preprocessing
   ✓ Implement ensemble methods for robustness
   ✓ Regular retraining with validated data

4. MLFLOW BEST PRACTICES:
   ✓ Tag runs with data version (DVC + Git commit)
   ✓ Log data quality metrics alongside model metrics
   ✓ Store data validation reports as artifacts
   ✓ Compare models trained on different data versions
""")

print("\n🔗 View full experiment details in MLflow UI:")
print(f"   {TRACKING_URI}")
print(f"   Experiment: {EXPERIMENT_NAME}")
print("\n✅ Analysis complete!")