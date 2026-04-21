# Low-Level Design Document

## 1. API Endpoint Definitions

### 1.1 Health Check
- **Endpoint**: `GET /health`
- **Description**: Check API and model health status
- **Input**: None
- **Output**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "scaler_loaded": true,
    "uptime_seconds": 3600.5
  }
  ```
- **Status Codes**: 200 OK

### 1.2 Readiness Check
- **Endpoint**: `GET /ready`
- **Description**: Check if the API is ready to serve predictions
- **Input**: None
- **Output**: `{"status": "ready"}`
- **Status Codes**: 200 OK, 503 Service Unavailable (model not loaded)

### 1.3 Single Prediction
- **Endpoint**: `POST /predict`
- **Description**: Predict machine failure from a single sensor reading
- **Input**:
  ```json
  {
    "air_temperature": 300.0,
    "process_temperature": 310.0,
    "rotational_speed": 1500,
    "torque": 40.0,
    "tool_wear": 100,
    "product_type": "L"
  }
  ```
- **Validation**:
  - air_temperature: float, 250–350 K
  - process_temperature: float, 250–400 K
  - rotational_speed: int, ≥ 0
  - torque: float, ≥ 0
  - tool_wear: int, ≥ 0
  - product_type: string, one of "H", "M", "L"
- **Output**:
  ```json
  {
    "prediction": 0,
    "failure_probability": 0.0342,
    "risk_level": "LOW",
    "model_version": "1.0"
  }
  ```
- **Status Codes**: 200 OK, 422 Validation Error, 503 Model Not Loaded

### 1.4 Batch Prediction
- **Endpoint**: `POST /predict/batch`
- **Description**: Predict for multiple sensor readings
- **Input**: `{"readings": [<SensorInput>, ...]}`
- **Output**:
  ```json
  {
    "predictions": [<PredictionResponse>, ...],
    "total": 100,
    "failures_detected": 5
  }
  ```

### 1.5 Drift Detection
- **Endpoint**: `POST /drift/check`
- **Description**: Check data drift on a batch of readings
- **Input**: `{"readings": [<SensorInput>, ...]}`
- **Output**:
  ```json
  {
    "overall_drift": false,
    "n_drifted": 0,
    "total_features_checked": 10,
    "drifted_features": [],
    "features": {
      "Air temperature [K]": {
        "ks_statistic": 0.05,
        "ks_p_value": 0.85,
        "psi": 0.02,
        "drift_detected": false
      }
    }
  }
  ```

### 1.6 Retrain Trigger
- **Endpoint**: `POST /retrain?reason=drift_detected`
- **Description**: Trigger model retraining
- **Input**: Query parameter `reason` (string)
- **Output**:
  ```json
  {
    "status": "triggered",
    "message": "Retraining pipeline has been triggered.",
    "triggered_by": "drift_detected"
  }
  ```

### 1.7 Model Info
- **Endpoint**: `GET /model/info`
- **Description**: Get loaded model metadata and test metrics
- **Output**: Model version, test metrics, classification report

### 1.8 Prometheus Metrics
- **Endpoint**: `GET /metrics`
- **Description**: Prometheus-compatible metrics export
- **Output**: Plain text in Prometheus exposition format

## 2. Module Specifications

### 2.1 src/data_ingestion.py
- `load_raw_data(filepath) → DataFrame`
- `validate_schema(df) → bool`
- `validate_data_quality(df) → dict`
- `split_data(df, test_size, random_state) → (DataFrame, DataFrame)`
- `run_ingestion(config) → dict`

### 2.2 src/data_preprocessing.py
- `engineer_features(df) → DataFrame`
- `get_feature_columns() → list[str]`
- `fit_scaler(df, feature_cols, save_path) → StandardScaler`
- `apply_scaler(df, feature_cols, scaler) → DataFrame`
- `compute_drift_baselines(df, feature_cols) → dict`
- `run_preprocessing(config) → dict`

### 2.3 src/model_training.py
- `evaluate_model(model, X, y) → dict`
- `train_model(X_train, y_train, X_test, y_test, config) → dict`
- `register_best_model(run_id, model_name, config) → str`
- `run_training(config) → dict`

### 2.4 src/drift_detection.py
- `ks_test(reference, current, threshold) → dict`
- `compute_psi(reference, current, n_bins) → float`
- `detect_drift(current_data, baselines, feature_cols, ...) → dict`

## 3. Data Schemas

### Raw Data (ai4i2020.csv)
| Column | Type | Description |
|--------|------|-------------|
| UDI | int | Unique identifier |
| Product ID | str | Product identifier |
| Type | str | Quality type (H/M/L) |
| Air temperature [K] | float | Ambient temperature |
| Process temperature [K] | float | Process temperature |
| Rotational speed [rpm] | int | Spindle speed |
| Torque [Nm] | float | Applied torque |
| Tool wear [min] | int | Cumulative tool usage |
| Machine failure | int | Target variable (0/1) |
| TWF–RNF | int | Failure type flags |

### Engineered Features
| Feature | Formula |
|---------|---------|
| temp_diff | Process temp − Air temp |
| power | Torque × RPM × 2π/60 |
| wear_degree | Tool wear × Torque |
| speed_torque_ratio | RPM / Torque |
| type_encoded | H=0, L=1, M=2 |
