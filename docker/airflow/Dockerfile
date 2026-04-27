# syntax=docker/dockerfile:1.7
# Airflow — Predictive Maintenance Pipeline

ARG AIRFLOW_VERSION=2.9.2
ARG PYTHON_VERSION=3.11

FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}

ARG AIRFLOW_VERSION
ARG PYTHON_VERSION

ENV AIRFLOW_HOME=/opt/airflow \
    PYTHONPATH=/opt/airflow:/app \
    PIP_NO_CACHE_DIR=1

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc curl git \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# Install extras using Airflow's constraint file
COPY --chown=airflow:root docker/requirements/airflow.txt /tmp/airflow-extras.txt
RUN pip install --no-cache-dir -r /tmp/airflow-extras.txt \
        --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# Copy project source
COPY --chown=airflow:root src/ /app/src/
COPY --chown=airflow:root configs/ /app/configs/
COPY --chown=airflow:root airflow/dags/ /opt/airflow/dags/

WORKDIR /opt/airflow
