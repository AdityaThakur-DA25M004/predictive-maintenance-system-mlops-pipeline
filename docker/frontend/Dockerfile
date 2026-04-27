# syntax=docker/dockerfile:1.7
# Streamlit Frontend — Predictive Maintenance
# ALWAYS runs on port 8501 inside the container.
# docker-compose maps FRONTEND_PORT (e.g. 3000) → 8501.

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEXSRFPROTECTION=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --uid 1000 app

WORKDIR /app

COPY docker/requirements/frontend.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=app:app frontend/ /app/frontend/
COPY --chown=app:app docs/user_manual.md /app/docs/user_manual.md

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# ALWAYS use 8501 — never use FRONTEND_PORT inside the container
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
