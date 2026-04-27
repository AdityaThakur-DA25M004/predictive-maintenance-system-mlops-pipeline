# Test Plan

> Predictive Maintenance System — test strategy, coverage map, acceptance criteria.

## 1. Scope

This plan covers automated testing for the Predictive Maintenance ML Pipeline. It does **not** cover load testing, security penetration testing, or chaos engineering — those are out of scope for the academic deliverable.

## 2. Test types and coverage

| Layer | Type | Files | Count |
|---|---|---|---|
| Data ingestion | Unit | `tests/test_data_ingestion.py` | 11 |
| Preprocessing & feature engineering | Unit | `tests/test_preprocessing.py` | 6 |
| Drift detection (KS + PSI) | Unit | `tests/test_drift.py` | 4 |
| Model evaluation | Unit / Integration | `tests/test_model.py` | 3 |
| API endpoints | Integration | `tests/test_api.py` | 18 |
| **Total** | | | **42** |

## 3. Test framework and tooling

- **pytest** — primary test runner.
- **pytest fixtures** — shared `sample_data`, `sample_reading`, `client`, `trained_model` fixtures via `conftest.py`.
- **fastapi.testclient.TestClient** — for API integration tests (no live server required).
- **tmp_path** — pytest's temp-directory fixture for filesystem-touching tests.

## 4. Acceptance criteria

The build is acceptable for delivery / demo if:

| # | Criterion | How verified |
|---|---|---|
| 1 | All 42 unit + integration tests pass | `pytest -q` exit code 0 |
| 2 | Best model F1 ≥ 0.80 | `models/test_metrics.json` |
| 3 | Best model ROC-AUC ≥ 0.90 | `models/test_metrics.json` |
| 4 | Single-prediction p99 latency < 200 ms | Prometheus histogram on `predict_latency_ms` |
| 5 | Drift on identical distributions returns `overall_drift=False` | `test_drift.test_identical_distributions_no_drift` |
| 6 | Drift on shifted distributions returns `overall_drift=True` | `test_drift.test_shifted_distribution_detects_drift` |
| 7 | Leaky columns (TWF/HDF/PWF/OSF/RNF) are dropped before training | `test_data_ingestion.test_removes_failure_mode_cols` |
| 8 | API returns 401/403 on `/retrain*` without valid key | `test_api.test_retrain_without_api_key`, `test_retrain_with_wrong_key` |
| 9 | Schema-invalid uploads rejected with 4xx | `test_data_ingestion.test_missing_columns` + `test_api.test_predict_invalid_product_type` |
| 10 | Prometheus `/metrics` endpoint serves valid exposition | `test_api.test_prometheus_metrics` |

## 5. Coverage map by component

### 5.1 `data_ingestion.py`
- ✅ Loading existing & missing files
- ✅ Schema validation (valid + missing columns)
- ✅ Data quality report (pass + missing-value detection)
- ✅ Stratified split sizes & class ratio preservation
- ✅ Leaky column removal (failure-mode flags, identifiers)
- ✅ Feature column preservation

### 5.2 `data_preprocessing.py`
- ✅ Feature engineering: derived features added (`temp_diff`, `power`, `wear_degree`, `type_encoded`, `speed_torque_ratio`)
- ✅ `temp_diff` arithmetic correctness
- ✅ Type encoding correctness (H=0, L=1, M=2)
- ✅ Idempotence — no mutation of input DataFrame
- ✅ Scaler fit + apply round-trip
- ✅ Drift baselines structure (mean/std/min/max/quantiles per feature)

### 5.3 `drift_detection.py`
- ✅ KS test on identical distributions → no drift
- ✅ KS test on shifted distributions → drift
- ✅ PSI on identical data → low value
- ✅ PSI on shifted data → high value (>0.2)

### 5.4 `model_training.py`
- ✅ Returns full metric set
- ✅ All metrics are within valid range [0, 1]
- ✅ Confusion matrix sums equal test-set size

### 5.5 API (`main.py`)
- ✅ Health & readiness — with and without model loaded
- ✅ Model info endpoint
- ✅ Prediction — valid input, invalid type, missing field, out-of-range temperature
- ✅ Batch prediction — happy path + empty input rejection
- ✅ Feedback — valid submission + unknown prediction ID
- ✅ Feedback stats endpoint
- ✅ Retrain — without key (401), wrong key (403), valid key (200)
- ✅ Drift check — empty payload rejected
- ✅ Prometheus `/metrics` endpoint

## 6. How to run

### Local (Windows PowerShell, repo root)
```powershell
# Activate venv first
.\venv\Scripts\Activate.ps1

# Full test suite
pytest -q

# With coverage report
pytest --cov=src --cov=api --cov-report=term-missing

# Single file
pytest tests/test_drift.py -v

# Single test
pytest tests/test_drift.py::TestKSTest::test_identical_distributions_no_drift -v
```

### In Docker
```powershell
docker compose exec api pytest -q
```

## 7. Test environment

- Python 3.11
- Tests run with synthetic and small-fixture data — no external network calls.
- API tests use `TestClient`, not a live HTTP server.
- No reliance on `mlflow:5000`, `airflow-webserver:8080`, or DagsHub during test execution.

## 8. Out-of-scope (deliberate)

- **Load testing** — single-replica RandomForest is the bottleneck, not the API; would need a separate load test harness (e.g., Locust) that's outside the assignment scope.
- **End-to-end browser testing** — UI is Streamlit, would require Selenium or Playwright; visual smoke testing handled manually during demo.
- **Security testing** — covered conceptually via API-key gating; no formal pentest.
- **Chaos / failure injection** — not part of the academic deliverable.

## 9. Continuous integration

Tests are intended to be run:
- On every developer push (ad-hoc, locally).
- Before any `dvc repro` invocation that produces a release-tagged version.
- Before promoting a model to production via the rollback API.

CI hosting (e.g., GitHub Actions) is **not** configured — the assignment forbids cloud-hosted CI. DVC's reproducibility (`dvc repro` honouring file hashes) provides the equivalent guarantee on a developer machine.
