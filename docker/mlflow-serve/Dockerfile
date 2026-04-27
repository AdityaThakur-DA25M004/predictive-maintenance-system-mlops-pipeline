# ============================================================
# Dockerfile for the mlflow-serve sidecar
# ============================================================
# WHY A SEPARATE IMAGE
# --------------------
# The model registered in MLflow was trained with these exact
# dependency versions (logged automatically by mlflow.sklearn.log_model
# at training time, embedded in the model's MLmodel descriptor):
#
#   mlflow==2.13.2
#   scikit-learn==1.4.2
#   cloudpickle==2.2.1
#   packaging==24.0
#   psutil==5.9.8
#   pyyaml==6.0.1
#
# The pm-api container ships *newer* versions of these (mlflow 2.22.4,
# sklearn 1.8.0, etc.) which is fine for the FastAPI service that
# loads the joblib directly — joblib is forgiving about minor sklearn
# version drift if the model class still exists.
#
# But `mlflow models serve` is much stricter:
#   1. It loads the embedded MLflow serving wrapper, which is
#      version-locked. Wrapper logged by mlflow 2.13.2 expects
#      gunicorn; mlflow 2.22.4 invokes uvicorn instead → crashes.
#   2. It deserializes via cloudpickle. cloudpickle 3.1.2 cannot
#      always read pickles written by 2.2.1.
#   3. The InconsistentVersionWarning from sklearn 1.8 vs 1.4.2 means
#      decision tree internals MAY have moved, so predictions become
#      unreliable.
#
# Pinning to the trained-against versions removes all three issues.
# This is also the textbook MLOps pattern: the serving environment
# must replicate the training environment.
# ============================================================

FROM python:3.11-slim

# curl is needed for the healthcheck (GET /ping in compose)
# build-essential is sometimes needed for transitive C extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pin EVERY dependency the trained model expects. Order matters:
# install mlflow last so it picks up the already-resolved deps below
# instead of bringing in newer transitives.
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

# Default working directory inside the container. mlflow models serve
# does not write to disk during inference, but having a writable cwd
# helps if MLflow logs locally for any reason.
WORKDIR /app

# Expose the serving port (matches docker-compose.mlflow-serve.yml)
EXPOSE 5001

# Note: the actual `mlflow models serve` command is invoked from the
# compose file's `command:` directive, not here. That keeps the image
# generic — the same image could serve different model URIs.