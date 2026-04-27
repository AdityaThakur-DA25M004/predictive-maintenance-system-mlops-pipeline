# Dockerfile for the mlflow-serve sidecar
# WHY A SEPARATE IMAGE
# The model registered in MLflow was trained with these exact
# dependency versions (logged automatically by mlflow.sklearn.log_model
# at training time, embedded in the model's MLmodel descriptor):

FROM python:3.11-slim

# curl is needed for the healthcheck (GET /ping in compose)
# build-essential is sometimes needed for transitive C extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir \
        scikit-learn==1.4.2 \
        cloudpickle==2.2.1 \
        packaging==24.0 \
        psutil==5.9.8 \
        pyyaml==6.0.1 \
        numpy \
        pandas \
        joblib \
        scipy \
        gunicorn \
        mlflow==2.13.2

WORKDIR /app

# Expose the serving port (matches docker-compose.mlflow-serve.yml)
EXPOSE 5001
