# =============================================================================
# Activate local Python environment for the predictive-maintenance project.
# =============================================================================
# Usage:
#   .\scripts\activate-local.ps1
#
# After this:
#   python -m src.data_ingestion           # uses local config + DagsHub MLflow
#   uvicorn src.api.main:app --reload      # local API on http://127.0.0.1:8000
#   streamlit run frontend/app.py          # local frontend on http://localhost:8501
#   dvc repro                              # uses .dvc/config.local for credentials
#   dvc push                               # pushes to DagsHub
#
# Docker is unaffected — these env vars only exist in this PowerShell session.
# =============================================================================

#Requires -Version 5.1
$ErrorActionPreference = "Stop"

# 1. cd to project root (script lives in scripts/, parent is project root)
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $projectRoot "configs"))) {
    $projectRoot = (Get-Location).Path
}
Set-Location $projectRoot

# 2. Activate venv
$venvActivate = Join-Path $projectRoot "venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Host "venv not found at $venvActivate" -ForegroundColor Red
    Write-Host "Create one first:" -ForegroundColor Yellow
    Write-Host "  python -m venv venv" -ForegroundColor Yellow
    exit 1
}
& $venvActivate

# 3. Set env vars that override Docker defaults
$env:DOTENV_PATH    = ".env.local"
$env:PM_CONFIG_PATH = "configs/config.local.yaml"
$env:PYTHONPATH     = "."

# 4. Sanity-check the local files exist
$missing = @()
if (-not (Test-Path $env:DOTENV_PATH))    { $missing += $env:DOTENV_PATH }
if (-not (Test-Path $env:PM_CONFIG_PATH)) { $missing += $env:PM_CONFIG_PATH }
if ($missing.Count -gt 0) {
    Write-Host "WARNING: missing local config files:" -ForegroundColor Yellow
    foreach ($f in $missing) { Write-Host "  - $f" -ForegroundColor Yellow }
    Write-Host "Code will fall back to Docker defaults." -ForegroundColor Yellow
    Write-Host ""
}

# 5. Echo what changed
$q = [char]39
Write-Host ""
Write-Host "Local environment activated:" -ForegroundColor Green
Write-Host "  Project root   : $projectRoot"
Write-Host "  venv           : $env:VIRTUAL_ENV"
Write-Host "  DOTENV_PATH    : $env:DOTENV_PATH"
Write-Host "  PM_CONFIG_PATH : $env:PM_CONFIG_PATH"
Write-Host "  PYTHONPATH     : $env:PYTHONPATH"
Write-Host ""
Write-Host "Try:"
Write-Host "  python -c `"from src.utils import load_config; print(load_config()[${q}mlflow${q}][${q}tracking_uri${q}])`""
Write-Host "  dvc status"
Write-Host "  dvc repro"
function dvc { python -m dvc $args }
