# User Manual — Predictive Maintenance System

## 1. Introduction

The **Predictive Maintenance System** monitors industrial equipment using sensor data to predict machine failures before they occur. Early detection reduces unplanned downtime, avoids safety risks, and lowers maintenance costs.

The model was trained on the AI4I 2020 dataset (10,000 records, 5 failure types) and is served as a REST API behind a Streamlit dashboard.

---

## 2. Getting Started

### 2.1 System Requirements
- A modern web browser (Chrome, Firefox, Edge)
- Network access to the application server
- No software installation required

### 2.2 Accessing the Dashboard
Open your browser and navigate to `http://<server-address>:8501`.

---

## 3. Making Predictions

### 3.1 Single Prediction
1. Click **🔮 Predict** in the sidebar.
2. Enter sensor readings:

   | Field | Description | Typical range |
   |-------|-------------|---------------|
   | Air Temperature (K) | Ambient air temperature | 295–305 K |
   | Process Temperature (K) | Machine process temperature | 305–315 K |
   | Rotational Speed (RPM) | Spindle speed | 1,000–2,500 RPM |
   | Torque (Nm) | Applied torque | 20–80 Nm |
   | Tool Wear (min) | Cumulative tool usage time | 0–250 min |
   | Product Type | Quality category — H, M, or L | — |

3. Click **🔍 Predict**.
4. Review the result: prediction, failure probability, risk level, and **prediction ID**.

### 3.2 Batch Prediction
1. Open the **Batch Prediction** tab on the Predict page.
2. Prepare a CSV with the required columns.
3. Upload the file and click **🚀 Run Batch Prediction**.
4. Download or review the results table.

### 3.3 Understanding Risk Levels

| Risk Level | Probability | Recommended Action |
|------------|-------------|--------------------|
| 🟢 LOW | < 20% | Normal operation — continue monitoring |
| 🟡 MEDIUM | 20–50% | Schedule maintenance within the next shift |
| 🟠 HIGH | 50–80% | Prioritize inspection before next run |
| 🔴 CRITICAL | > 80% | Stop machine immediately, inspect now |

---

## 4. Closing the Loop — Submitting Ground Truth

After a prediction, once you observe whether the machine truly failed:

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"prediction_id": 42, "actual_label": 1}'
```

The rolling accuracy is updated immediately and displayed in Grafana.

---

## 5. Monitoring

Navigate to **📊 Monitoring** to view model performance, run drift detection, and trigger retraining.

---

## 6. Failure Types Explained

| Code | Type | Cause |
|------|------|-------|
| TWF | Tool Wear Failure | Tool used beyond safe wear limit |
| HDF | Heat Dissipation Failure | Temperature differential too high |
| PWF | Power Failure | Product of torque × speed outside bounds |
| OSF | Overstrain Failure | Excessive torque for the product type |
| RNF | Random Failure | Stochastic failure (~0.1% probability) |

---

## 7. Troubleshooting

| Issue | Solution |
|-------|----------|
| "API Offline" on dashboard | Ensure containers are running: `docker compose ps` |
| Model not loaded | Run the training pipeline or check MLflow |
| Drift detected | Review drifted features; consider triggering retraining |
| 401 on `/retrain` | Set `X-API-Key` header with value of `RETRAIN_API_KEY` |

---

## 8. Advanced Access

| Dashboard | URL |
|-----------|-----|
| MLflow UI | http://localhost:5000 |
| Airflow UI | http://localhost:8080 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| FastAPI docs | http://localhost:8000/docs |
