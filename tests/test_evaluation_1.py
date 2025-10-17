import pandas as pd
import pytest
import dvc.api
import pandera as pa
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# --- Schema and Fixtures (No Changes Needed) ---

iris_schema = pa.DataFrameSchema({
    "sepal_length": pa.Column(float, pa.Check.in_range(4.0, 8.0)),
    "sepal_width": pa.Column(float, pa.Check.in_range(2.0, 5.0)),
    "petal_length": pa.Column(float, pa.Check.in_range(1.0, 7.0)),
    "petal_width": pa.Column(float, pa.Check.in_range(0.0, 3.0)),
    "species": pa.Column(str, pa.Check.isin(['setosa', 'versicolor', 'virginica']))
})

@pytest.fixture(scope="module")
def dvc_data():
    """Fixture to pull data from DVC."""
    with dvc.api.open('data/data.csv', mode='r', encoding='utf-8') as f:
        return pd.read_csv(f)

@pytest.fixture(scope="module")
def dvc_model():
    """Fixture to pull the model from DVC."""
    with dvc.api.open('model/model_joblib.pkl', mode='rb') as f:
        return joblib.load(f)

# --- New Function to Generate the CML Report ---

def generate_cml_report(y_true, y_pred, accuracy):
    """Generates a markdown report with metrics and plots."""
    with open("cml_report.md", "w") as f:
        f.write("## Model Performance Report\n\n")
        f.write(f"**Overall Accuracy:** `{accuracy:.4f}`\n\n")
        
        # --- Classification Metrics ---
        report = classification_report(y_true, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        f.write("### Classification Metrics\n\n")
        f.write(df_report.to_markdown())
        f.write("\n\n")

        # --- Confusion Matrix Plot ---
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap='Blues')
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        
        f.write("### Confusion Matrix\n\n")
        f.write("![Confusion Matrix](confusion_matrix.png)")

# --- Updated Test Functions ---

def test_data_validation(dvc_data):
    """Test data validation using pandera schema."""
    print("\n--- Data Validation ---")
    try:
        iris_schema.validate(dvc_data, lazy=True)
        print("✅ Data validation passed!")
    except pa.errors.SchemaErrors as err:
        print("❌ Data validation failed!")
        print(err)
        pytest.fail(f"Data validation failed: {err}")

def test_model_evaluation_and_report(dvc_data, dvc_model):
    """Test model evaluation and generate the CML report."""
    print("\n--- Model Evaluation ---")
    X = dvc_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_true = dvc_data['species']

    # Get model predictions
    predictions = dvc_model.predict(X)
    accuracy = accuracy_score(y_true, predictions)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Assert that the model accuracy is above the threshold
    assert accuracy > 0.95, f"Model accuracy is too low: {accuracy:.4f}"
    print("✅ Model accuracy check passed!")

    # If the assertion passes, generate the full report
    print("\n--- Generating CML Report Artifacts ---")
    generate_cml_report(y_true, predictions, accuracy)
    print("✅ Report artifacts generated successfully.")