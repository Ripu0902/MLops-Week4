# Data Poisoning Study Results

## 📁 Files Generated

- `poisoning_analysis_dashboard.png` - Comprehensive visualization of poisoning impact
- `poisoning_summary.csv` - Tabular summary of all experiment runs
- `confusion_matrix_*.png` - Confusion matrices for each poisoning level
- `cml_report_poisoning.md` - Markdown report for CI/CD integration
- `metrics_poisoning.json` - JSON metrics for automated processing
- `data_validation_report.json` - Data quality assessment results
- `robustness_metrics.json` - Model robustness test results

## 🔗 View in MLflow

Access the MLflow UI to explore all experiment runs interactively:
- Experiment: `iris_data_poisoning`
- Compare runs side-by-side
- Download models and artifacts
- View detailed metrics and parameters

## 📊 Key Findings

Review `poisoning_analysis_dashboard.png` for visual insights into:
1. How accuracy degrades with increasing poisoning levels
2. Comparison between feature and label poisoning
3. Train vs test accuracy gaps indicating data quality issues
4. F1 score trends across poisoning levels

## 🛡️ Next Steps

1. Review the validation test results
2. Check if any models failed quality thresholds
3. Document lessons learned
4. Update production data validation pipelines
