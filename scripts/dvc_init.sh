#!/usr/bin/env bash
# ============================================================
# DVC + DagsHub bootstrap
# ============================================================
set -euo pipefail

log()  { echo "[dvc_init] $*"; }
warn() { echo "[dvc_init] WARN: $*" >&2; }

cd "${WORKDIR:-/app}"

# 1. Git
if [ ! -d ".git" ]; then
    log "No .git directory found — initialising empty git repo"
    git init -q
    git config user.email "dvc@predictive-maintenance.local"
    git config user.name  "DVC Runner"
    git add -A || true
    git commit -qm "chore: initial commit for DVC" || true
else
    log ".git already present"
    git config --global --add safe.directory "${WORKDIR:-/app}" || true
fi

# 2. DVC
if [ ! -d ".dvc" ]; then
    log "Initialising DVC"
    # SCM-aware init is preferred since we ensured .git exists in step 1.
    # --no-scm is only a fallback for environments where git init failed.
    dvc init -q || dvc init -q --no-scm
else
    log ".dvc already present"
fi

# 3. Remote
REMOTE_NAME="${DVC_REMOTE_NAME:-dagshub}"
REMOTE_URL="${DVC_REMOTE_URL:-s3://dvc}"
REMOTE_ENDPOINT="${DVC_REMOTE_ENDPOINT:-}"
ACCESS_KEY="${DVC_ACCESS_KEY_ID:-${DAGSHUB_USERNAME:-}}"
SECRET_KEY="${DVC_SECRET_ACCESS_KEY:-${DAGSHUB_TOKEN:-}}"

if [ -z "$REMOTE_ENDPOINT" ] || [ -z "$ACCESS_KEY" ] || [ -z "$SECRET_KEY" ]; then
    warn "DVC remote credentials missing — skipping remote setup"
else
    log "Configuring DVC remote '$REMOTE_NAME' → $REMOTE_ENDPOINT"
    if dvc remote list | grep -q "^${REMOTE_NAME}\b"; then
        dvc remote modify "$REMOTE_NAME" url "$REMOTE_URL"
    else
        dvc remote add -d "$REMOTE_NAME" "$REMOTE_URL"
    fi
    dvc remote modify --local "$REMOTE_NAME" endpointurl       "$REMOTE_ENDPOINT"
    dvc remote modify --local "$REMOTE_NAME" access_key_id     "$ACCESS_KEY"
    dvc remote modify --local "$REMOTE_NAME" secret_access_key "$SECRET_KEY"
    log "DVC remote configured successfully"
fi

if [ $# -eq 0 ]; then
    log "No command given, showing DVC status"
    exec dvc status
else
    log "Executing: $*"
    exec "$@"
fi