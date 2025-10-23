#!/bin/bash

mlflow server --backend-store-uri postgresql+psycopg2://mlflowuser:StrongPassword123@34.31.16.124:5432/mlflowdb --default-artifact-root gs://vertex-mlflow-artifacts-electric-wave-472614-d5 --host 0.0.0.0 --port 6969 --cors-allowed-origins "*" --allowed-hosts "*"
