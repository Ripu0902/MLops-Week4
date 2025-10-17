import pandas as pd
import pytest
import dvc.api
import pandera as pa
from sklearn.metrics import accuracy_score
import joblib
import io
import sys

# Define the expected schema for the Iris dataset
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
    with dvc.api.open('data/data.csv', mode='rb') as f:
        return pd.read_csv(f)

@pytest.fixture(scope="module")
def dvc_model():
    """Fixture to pull the model from DVC."""
    with dvc.api.open('model/model_joblib.pkl', mode='rb') as f:
        return joblib.load(f)

def test_data_validation(dvc_data, capsys):
    """Test data validation using pandera schema."""
    with capsys.disabled():
        print("\n--- Data Validation ---")
    try:
        iris_schema.validate(dvc_data, lazy=True)
        with capsys.disabled():
            print("✅ Data validation passed!")
    except pa.errors.SchemaErrors as err:
        with capsys.disabled():
            print(f"❌ Data validation failed!")
            print(err)
        pytest.fail(f"Data validation failed: {err}")

def test_model_evaluation(dvc_data, dvc_model, capsys):
    """Test model evaluation and performance."""
    with capsys.disabled():
        print("\n--- Model Evaluation ---")
    X = dvc_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = dvc_data['species']

    predictions = dvc_model.predict(X)
    accuracy = accuracy_score(y, predictions)

    with capsys.disabled():
        print(f"Model accuracy: {accuracy:.4f}")
    
    # Assert that the model accuracy is above a certain threshold
    assert accuracy > 0.95, f"Model accuracy is too low: {accuracy:.4f}"

# Function to generate the CML report
def generate_cml_report(test_results):
    with open("cml_report.md", "w") as f:
        f.write("# Sanity Test Report\n\n")
        f.write("## Test Results\n\n")
        f.write("```\n")
        f.write(test_results)
        f.write("\n```\n")
        
        # Add a table for metrics if you have any
        # f.write("## Metrics\n\n")
        # f.write("| Metric | Value |\n")
        # f.write("|---|---|\n")
        # f.write(f"| Model Accuracy | `{accuracy:.4f}` |\n")

if __name__ == "__main__":
    # Capture pytest output
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    
    # Run tests and get results
    pytest.main([__file__])
    
    # Generate the CML report
    sys.stdout = old_stdout
    generate_cml_report(captured_output.getvalue())

