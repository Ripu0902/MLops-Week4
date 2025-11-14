# Model Evaluation Report — Data Poisoning Study

**Run ID:** `unknown`
**Model URI:** `gs://vertex-mlflow-artifacts-electric-wave-472614-d5/3/models/m-b0fe3525681d4c93bb616559e89d8b64/artifacts`

## Overall Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Accuracy | `0.6600` | ❌ FAIL |
| F1 Score | `0.6488` | ❌ FAIL |

## Per-Class Performance

| Class | Accuracy |
|-------|----------|
| setosa | `0.7600` |
| versicolor | `0.3200` |
| virginica | `0.9000` |

## Confusion Matrix

![Confusion Matrix](confusion_matrix_validation.png)

## ⚠️  Alerts

**Potential data poisoning or model degradation detected!**

- Model performance is below acceptable thresholds
- Review training data for anomalies
- Check data validation reports
- Consider retraining with clean data
