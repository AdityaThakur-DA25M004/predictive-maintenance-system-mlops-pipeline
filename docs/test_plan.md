# Test Plan & Test Cases

## 1. Testing Strategy

| Level | Scope | Tool |
|-------|-------|------|
| Unit | Individual functions (ingestion, preprocessing, drift, training) | pytest |
| Integration | API endpoints with TestClient | pytest + fastapi.testclient |
| End-to-end | Full Docker Compose stack | docker compose + curl |

## 2. Acceptance Criteria

1. All unit and integration tests pass (0 failures, 0 errors).
2. `GET /health` returns HTTP 200 in under 1 second.
3. `POST /predict` returns valid JSON with `prediction`, `failure_probability`, `risk_level`, `prediction_id`.
4. `POST /feedback` records ground truth and updates rolling accuracy.
5. Model F1 ≥ 0.65 on the AI4I 2020 hold-out set.
6. p95 inference latency < 200 ms (business SLO).
7. Drift detector flags synthetic drift in 100% of injected-drift test cases.
8. `docker compose up --build` brings all services to healthy state.
9. `/retrain` returns 401 without `X-API-Key` and 200 with a valid key.
10. Prometheus successfully scrapes `/metrics`.

## 3. Test Cases

### 3.1 Data Ingestion (`test_data_ingestion.py`)

| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| DI-01 | Load existing CSV file | DataFrame with 100 rows loaded |
| DI-02 | Load non-existent file | FileNotFoundError raised |
| DI-03 | Validate correct schema | Returns True |
| DI-04 | Validate schema with missing column | ValueError raised |
| DI-05 | Data quality on clean data | Status = PASS, 0 missing |
| DI-06 | Data quality with missing values | Detects missing count > 0 |
| DI-07 | Train/test split sizes | 80 / 20 |
| DI-08 | Stratified split preservation | Failure rate delta < 0.05 |
| DI-09 | Leaky columns dropped (failure modes) | TWF/HDF/PWF/OSF/RNF removed |
| DI-10 | Identifier columns dropped | UDI/Product ID removed |
| DI-11 | Features and target kept | Machine failure, Air temp, Type present |

### 3.2 Preprocessing (`test_preprocessing.py`)

| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| PP-01 | Feature engineering adds columns | 5 new features present |
| PP-02 | `temp_diff` calculation correct | Process − Air temp |
| PP-03 | Type encoding correct | H=0, L=1, M=2 |
| PP-04 | No input mutation | Original DataFrame unchanged |
| PP-05 | Scaler fit and transform | Mean ≈ 0 after scaling |
| PP-06 | Drift baselines structure | All stats keys present |

### 3.3 Drift Detection (`test_drift.py`)

| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| DR-01 | KS test on identical distributions | `drift_detected = False` |
| DR-02 | KS test on shifted distribution (+3σ) | `drift_detected = True` |
| DR-03 | PSI ≈ 0 for identical data | PSI < 0.1 |
| DR-04 | PSI > 0.25 on large shift | PSI > 0.25 |

### 3.4 Model (`test_model.py`)

| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| ML-01 | `evaluate_model` returns all metrics | All 9 keys present |
| ML-02 | Metrics in valid range | All between 0 and 1 |
| ML-03 | Confusion matrix sums to N | TP+TN+FP+FN = total |

### 3.5 API (`test_api.py`)

| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| API-01 | `GET /health` | 200, `status=healthy` |
| API-02 | `GET /ready` with model | 200, `status=ready` |
| API-03 | `GET /ready` without model | 503 |
| API-04 | `GET /model/info` | 200, `model_loaded` field |
| API-05 | `POST /predict` valid input | 200, prediction in {0,1}, `prediction_id` present |
| API-06 | `POST /predict` invalid product_type | 422 |
| API-07 | `POST /predict` missing fields | 422 |
| API-08 | `POST /predict` out-of-range temp | 422 |
| API-09 | `POST /predict/batch` valid | 200, correct total |
| API-10 | `POST /predict/batch` empty | 422 |
| API-11 | `POST /feedback` valid prediction_id | 200, `correct` bool returned |
| API-12 | `POST /feedback` unknown prediction_id | 404 |
| API-13 | `GET /feedback/stats` | 200, reports totals |
| API-14 | `POST /retrain` without X-API-Key | 401 |
| API-15 | `POST /retrain` with wrong key | 401 |
| API-16 | `POST /retrain` with valid key | 200, status=triggered |
| API-17 | `GET /metrics` | 200, contains `prediction_requests_total` |
| API-18 | `POST /drift/check` with empty batch | 422 |

## 4. Test Execution

```bash
# All tests
pytest tests/ -v --tb=short

# Coverage
pytest tests/ -v --cov=src --cov=api --cov-report=html --cov-report=term

# One module
pytest tests/test_api.py -v
```

## 5. Test Report

Results from validated test run:

| Metric | Value |
|--------|-------|
| Total test cases | 42 |
| Passed | 42 |
| Failed | 0 |
| Skipped | 0 |
| Test run time | 14.0 s |
| Date | 2026-04-20 |

### Model Acceptance

| Metric | Threshold | Observed |
|--------|-----------|----------|
| F1-score | ≥ 0.65 | **0.848** |
| ROC-AUC | ≥ 0.85 | **0.961** |
| Precision | — | **0.930** |
| Recall | — | **0.779** |
| Accuracy | — | **0.991** |

All acceptance criteria met.
