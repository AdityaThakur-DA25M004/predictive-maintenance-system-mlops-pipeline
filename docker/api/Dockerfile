# syntax=docker/dockerfile:1.7
# FastAPI Backend — Predictive Maintenance
# Multi-stage build to keep the runtime image small.

# ---- Stage 1: builder ----
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY docker/requirements/api.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip wheel --wheel-dir=/wheels -r requirements.txt


# ---- Stage 2: runtime ----
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    PATH="/home/app/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --uid 1000 app

WORKDIR /app

# Install pre-built wheels
COPY --from=builder /wheels /wheels
COPY docker/requirements/api.txt ./requirements.txt
RUN pip install --no-index --find-links=/wheels -r requirements.txt && \
    rm -rf /wheels

# Copy source
COPY --chown=app:app api/ /app/api/
COPY --chown=app:app src/ /app/src/
COPY --chown=app:app configs/ /app/configs/

# Writable directories
RUN mkdir -p /app/data/feedback /app/models /app/data/baselines && \
    chown -R app:app /app/data /app/models

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000} --workers ${API_WORKERS:-2} --log-level ${API_LOG_LEVEL:-info}"]
