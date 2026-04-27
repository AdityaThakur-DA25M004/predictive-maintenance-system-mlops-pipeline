# syntax=docker/dockerfile:1.7
# ============================================================
# DVC runner — one-shot container for `dvc repro` + `dvc push`
# ============================================================
# Purpose:
#   Reproduces the DVC pipeline (dvc.yaml stages) and pushes
#   data + model artifacts to DagsHub storage.
#
# Usage:
#   docker compose run --rm dvc dvc repro            # run pipeline
#   docker compose run --rm dvc dvc push             # upload outputs
#   docker compose run --rm dvc dvc status --cloud   # check sync
# ============================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

# DVC needs git and build-essential to install cleanly
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY docker/requirements/dvc.txt /tmp/dvc-requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/dvc-requirements.txt

# Copy bootstrap script — configures DagsHub remote on first run
COPY --chmod=0755 scripts/dvc_init.sh /usr/local/bin/dvc_init.sh

# Entry point wraps dvc_init.sh before any dvc command
ENTRYPOINT ["/usr/local/bin/dvc_init.sh"]
CMD ["dvc", "--help"]