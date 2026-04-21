"""
Predictive Maintenance — Main Dashboard.
"""

import streamlit as st
from frontend.common import APIError, get_client, render_header, render_sidebar, setup_page

setup_page("Dashboard", icon="🏭")
render_sidebar()
render_header("🏭 Predictive Maintenance Dashboard",
              "Real-time machine health monitoring & failure prediction")

client = get_client()

# ── System health ──────────────────────────────────────────────────────────
try:
    health = client.health()
    api_ok = True
except APIError:
    health = {}
    api_ok = False

c1, c2, c3, c4 = st.columns(4)
c1.metric("API Status", "🟢 Online" if api_ok else "🔴 Offline")
c2.metric("Model", "🟢 Loaded" if health.get("model_loaded") else "🔴 Not Loaded")
c3.metric("Scaler", "🟢 Loaded" if health.get("scaler_loaded") else "🔴 Not Loaded")
up = int(health.get("uptime_seconds", 0))
c4.metric("Uptime", f"{up // 3600}h {(up % 3600) // 60}m" if up >= 3600 else f"{up}s")

# ── Model performance ─────────────────────────────────────────────────────
st.markdown("### 📈 Model Performance")
try:
    info = client.model_info()
    metrics = info.get("test_metrics") or {}
except APIError:
    metrics = {}

if metrics:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    m2.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
    m3.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    m4.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    m5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
else:
    st.info("Model metrics not yet available. Run the training pipeline first.")

# ── Live feedback ─────────────────────────────────────────────────────────
try:
    fb = client.feedback_stats()
    if fb.get("total_feedback", 0) > 0:
        st.markdown("### 🔁 Live Feedback Accuracy")
        f1, f2, f3 = st.columns(3)
        f1.metric("Feedback received", fb["total_feedback"])
        ov = fb.get("overall_accuracy")
        rl = fb.get("rolling_accuracy")
        f2.metric("Overall accuracy", f"{ov:.1%}" if ov is not None else "—")
        f3.metric(f"Rolling (last {fb['window']})", f"{rl:.1%}" if rl is not None else "—")
except APIError:
    pass

# ── Overview ───────────────────────────────────────────────────────────────
st.markdown("### 🎯 System Overview")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
    **Purpose.** Predict industrial equipment failures *before* they happen
    using sensor data: temperature, speed, torque, tool wear, and product type.

    **Failure modes:** TWF (Tool Wear), HDF (Heat Dissipation),
    PWF (Power), OSF (Overstrain), RNF (Random)
    """)
with col_b:
    st.markdown("""
    **Architecture:**
    - **Backend:** FastAPI + Prometheus instrumentation
    - **ML:** Random Forest + MLflow tracking & registry
    - **Pipeline:** Airflow DAG with drift-triggered retraining
    - **Monitoring:** Prometheus + Grafana + alert rules
    - **Versioning:** DVC + DagsHub
    - **Deploy:** Docker Compose (8 containers)
    """)

st.info("**Next** · Use **🔮 Predict** to submit readings. "
        "Use **📊 Monitoring** to check drift. "
        "Use **🔄 Pipeline** to see the ML DAG.")
