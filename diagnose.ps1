# # ============================================================
# # DIAGNOSE — find the actual Streamlit error
# # ============================================================
# # Streamlit's "Connection error" dialog doesn't tell you WHY.
# # This script digs up the real error from logs + network.
# # ============================================================

# Write-Host "`n=== A. Full Streamlit container logs (last 200 lines) ===" -ForegroundColor Cyan
# docker compose logs --tail=200 frontend

# Write-Host "`n=== B. Is the Streamlit script itself crashing? ===" -ForegroundColor Cyan
# Write-Host "Running the exact streamlit command manually inside the container to see the error:"
# docker compose exec -T frontend bash -c "python -c 'import sys; sys.path.insert(0, \"/app\"); from frontend.common import get_client; print(\"import OK\")'" 2>&1

# Write-Host "`n=== C. What does the frontend directory actually look like inside container? ===" -ForegroundColor Cyan
# docker compose exec -T frontend ls -la /app/frontend/ 2>&1
# Write-Host ""
# docker compose exec -T frontend ls -la /app/frontend/pages/ 2>&1

# Write-Host "`n=== D. Does Streamlit's WebSocket endpoint respond? ===" -ForegroundColor Cyan
# # Try to hit the WebSocket URL directly using curl
# # A proper WebSocket upgrade would return 101; any other response is the clue
# try {
#     $r = curl.exe -sv --max-time 5 "http://localhost:3000/_stcore/stream" 2>&1 | Select-String -Pattern "HTTP|< "
#     $r | ForEach-Object { Write-Host $_ }
# } catch {
#     Write-Host "curl failed: $_" -ForegroundColor Red
# }

# Write-Host "`n=== E. Browser-side: open DevTools on localhost:3000 ===" -ForegroundColor Yellow
# Write-Host "  1. Press F12"
# Write-Host "  2. Go to the Console tab"
# Write-Host "  3. Refresh the page"
# Write-Host "  4. Look for lines starting with 'WebSocket connection to'"
# Write-Host "  5. Copy those lines — THAT is the real error"

# Write-Host "`n=== F. Check bind mount status (THE real issue) ===" -ForegroundColor Cyan
# Write-Host "Host:"
# Get-ChildItem data\raw | Format-Table Name, Length
# Write-Host "Inside airflow-scheduler container (should show the same file):"
# docker compose exec -T airflow-scheduler ls -la /app/data/raw/ 2>&1
# Write-Host ""
# Write-Host "Inside airflow-scheduler — check if /opt/airflow gets shadowed:"
# docker compose exec -T airflow-scheduler ls -la /opt/airflow/ 2>&1 | Select-Object -First 10

# Write-Host "`n=== Done. The 'real' Streamlit error is in section A or E ===" -ForegroundColor Green


# ============================================================
# APPLY FIXES — fixes the 2 remaining issues
# ============================================================
# Run from project root after copying the new files:
#   docker-compose.yml
#   docker/airflow/Dockerfile
#   docker/frontend/Dockerfile
#   docker/requirements/airflow.txt
#   docker/requirements/airflow-ml.txt
# ============================================================

Write-Host "`n=== STEP 1: Stop everything and remove airflow images ===" -ForegroundColor Cyan
docker compose down

# Remove the cached airflow images so they're FORCED to rebuild
Write-Host "`n=== STEP 2: Remove cached airflow + frontend images ===" -ForegroundColor Cyan
docker rmi predictive-maintenance-airflow-scheduler -f 2>$null
docker rmi predictive-maintenance-airflow-webserver -f 2>$null
docker rmi predictive-maintenance-airflow-init      -f 2>$null
docker rmi predictive-maintenance-frontend          -f 2>$null

# Build fresh — this WILL install sklearn etc. into Airflow this time
Write-Host "`n=== STEP 3: Rebuild airflow + frontend (no cache) ===" -ForegroundColor Cyan
Write-Host "    This takes 5-10 min — it's installing the ML stack." -ForegroundColor Yellow
docker compose build --no-cache airflow-webserver frontend
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Build failed. Check error above." -ForegroundColor Red
    exit 1
}

Write-Host "`n=== STEP 4: Start the stack ===" -ForegroundColor Cyan
docker compose up -d
Start-Sleep -Seconds 15

Write-Host "`n=== STEP 5: Verify sklearn is now installed in Airflow ===" -ForegroundColor Cyan
$result = docker compose exec -T airflow-scheduler python -c "import sklearn, mlflow, dvc; print('sklearn:', sklearn.__version__); print('mlflow:', mlflow.__version__); print('dvc:', dvc.__version__)" 2>&1
Write-Host $result
if ($result -match "sklearn:") {
    Write-Host "✅ ML stack installed successfully" -ForegroundColor Green
} else {
    Write-Host "❌ ML stack still missing — paste this output back" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== STEP 6: Test ingestion ===" -ForegroundColor Cyan
docker compose exec -T airflow-scheduler python -m src.data_ingestion 2>&1 | Select-Object -Last 5

Write-Host "`n=== STEP 7: Verify Streamlit serves correct browser port ===" -ForegroundColor Cyan
# Check what port Streamlit tells the browser to use
$streamlitConfig = curl.exe -sf "http://localhost:3000/_stcore/host-config" 2>&1 | Out-String
Write-Host "Streamlit host-config response:"
Write-Host $streamlitConfig

Write-Host "`n=== STEP 8: Trigger DAG ===" -ForegroundColor Cyan
$apiKey = (Select-String -Path .env -Pattern "RETRAIN_API_KEY=" | Select-Object -First 1).ToString().Split("=")[1]
curl.exe -X POST "http://localhost:8000/retrain?reason=manual" -H "X-API-Key: $apiKey"

Write-Host "`n`n=== Done. Now: ===" -ForegroundColor Green
Write-Host "  1. Refresh http://localhost:3000 — Streamlit should load (no 'Connection error')" -ForegroundColor Cyan
Write-Host "  2. Open http://localhost:8080 — watch DAG run end-to-end (admin/admin)" -ForegroundColor Cyan
Write-Host "  3. Open http://localhost:5000 — should see new MLflow runs in 'predictive-maintenance' experiment" -ForegroundColor Cyan