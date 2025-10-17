# MLOps Week 4 Graded Assignment 4

**Name:** Ripunjay Kumar  
**Roll No:** 21f3002511  
**Term:** Sept 2025  
**Course:** MLOps (BSDA5014)  
**Lab Done by:** Ripunjay Kumar

# ML Pipeline — CI/CD + Tests + CML Reports

Summary
- This repo runs automated tests that validate data, evaluate a trained model, and produce CML report artifacts (markdown + confusion matrix image).
- CI behavior is defined in the workflow file: [ci-cd.yml](ci-cd.yml).
- The main test implementation is in [test_evaluation_1.py](test_evaluation_1.py).


Quick links
- Workflow: [ci-cd.yml](ci-cd.yml)
- Tests & report generation: [test_evaluation_1.py](test_evaluation_1.py)
  - Test functions: [`test_data_validation`](test_evaluation_1.py), [`test_model_evaluation_and_report`](test_evaluation_1.py)
  - Fixtures: [`dvc_data`](test_evaluation_1.py), [`dvc_model`](test_evaluation_1.py)
  - Report generator: [`generate_cml_report`](test_evaluation_1.py)
- Existing generated reports:
  - [reports/1/merge-2025-10-17_18-57-54/cml_report.md](reports/1/merge-2025-10-17_18-57-54/cml_report.md)
  - [reports/2/merge-2025-10-17_19-34-33/cml_report.md](reports/2/merge-2025-10-17_19-34-33/cml_report.md)
  - [reports/dev-2025-10-17_18-54-53/cml_report.md](reports/dev-2025-10-17_18-54-53/cml_report.md)
  - [reports/dev-2025-10-17_19-36-46/cml_report.md](reports/dev-2025-10-17_19-36-46/cml_report.md)
  - [reports/main-2025-10-17_18-59-46/cml_report.md](reports/main-2025-10-17_18-59-46/cml_report.md)
- This README: [README.md](README.md)

What the tests do
- `test_data_validation` validates the input CSV against a Pandera schema (see schema in [test_evaluation_1.py](test_evaluation_1.py)).
- `test_model_evaluation_and_report`:
  - Loads model/data via DVC (paths used: `data/data.csv` and `model/model_joblib.pkl` inside [test_evaluation_1.py](test_evaluation_1.py)).
  - Computes predictions and accuracy, asserts accuracy > 0.95, and on success calls [`generate_cml_report`](test_evaluation_1.py) to write `cml_report.md` and `confusion_matrix.png`.

How CI runs (high level)
- The workflow [ci-cd.yml](ci-cd.yml) triggers on pushes to `dev` and `main` and on PRs.
- Steps include: checkout, set up Python, install dependencies, authenticate to GCP (uses `GCP_SA_KEY` secret), DVC pull, run tests, setup CML, post a CML comment, and upload report artifacts to GCS.
- Artifacts uploaded by CI: `cml_report.md` and `confusion_matrix.png` — see the generated reports linked above.

Run tests locally
1. Create a venv and install deps (CI installs via `ci-cd.yml`):
   - pip install -r req.txt pytest pandera scikit-learn pandas matplotlib joblib
2. Ensure DVC remote is available (same DVC config as CI) and run:
   - dvc pull
3. Run pytest:
   - pytest test_evaluation_1.py

Notes and troubleshooting
- CI expects a GCP service account JSON stored in the secret `GCP_SA_KEY` for `google-github-actions/auth@v2`.
- CI uses `dvc remote default gcsremote` then `dvc pull`. Ensure your DVC remotes and credentials are set up for local runs.
- The workflow in [ci-cd.yml](ci-cd.yml) runs `pytest tests/test_evaluation_1.py` — the repository currently contains `test_evaluation_1.py` at the repo root. If CI fails to find the test, either add a `tests/` folder or update the workflow path.
- Generated report artifacts (per-run) are stored in the `reports/` folder (examples linked above) and also uploaded to a GCS bucket by CI.

Useful symbols (for quick navigation)
- [`generate_cml_report`](test_evaluation_1.py)
- [`test_model_evaluation_and_report`](test_evaluation_1.py)
- [`test_data_validation`](test_evaluation_1.py)
- [`dvc_data`](test_evaluation_1.py)
- [`dvc_model`](test_evaluation_1.py)

License / contact
- See repo root for license or project policies.
- For CI failures, check the Actions log in GitHub and inspect upload/comment steps in [ci-cd.yml](ci-cd.yml).