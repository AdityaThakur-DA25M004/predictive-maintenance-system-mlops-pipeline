# syntax=docker/dockerfile:1.7
# ============================================================
# Trainer — unified ML + DVC + MLflow container
# ============================================================
# Used for TWO purposes:
#   1. Airflow spawns this via DockerOperator for each pipeline task
#      (data_ingestion, preprocessing, training, drift_check, dvc_push)
#   2. You run it manually for DVC commands:
#        docker compose run --rm trainer dvc repro
#        docker compose run --rm trainer dvc push
#        docker compose run --rm trainer dvc dag
#
# Has NO Airflow constraints so we can use modern versions of
# DVC, MLflow, and httpx without conflicts.
# ============================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --uid 1000 trainer

WORKDIR /app

# Install heavy ML + DVC stack — free to resolve without constraints
COPY docker/requirements/trainer.txt /tmp/trainer-requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/trainer-requirements.txt

# Copy DVC bootstrap script (configures DagsHub remote)
COPY --chmod=0755 scripts/dvc_init.sh /usr/local/bin/dvc_init.sh

# Project source — src + configs are also bind-mounted at runtime
# so code changes are live without rebuild
COPY src/ /app/src/
COPY configs/ /app/configs/

# Writable dirs owned by the trainer user
RUN mkdir -p /app/data /app/models && \
    chown -R trainer:trainer /app

# Allow git to operate on bind-mounted .git directory
RUN git config --system --add safe.directory /app

USER trainer

# Default command = show DVC status so the container is useful even
# when invoked with no args. Airflow DockerOperator overrides this.
ENTRYPOINT ["/usr/local/bin/dvc_init.sh"]
CMD ["dvc", "status"]