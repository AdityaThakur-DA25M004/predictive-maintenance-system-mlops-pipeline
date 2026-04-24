"""
Pipeline Visualization Page — End-to-end ML pipeline overview.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotly.graph_objects as go
import streamlit as st
from frontend.common import get_config, render_header, render_sidebar, setup_page

setup_page("Pipeline", icon="🔄")
render_sidebar()
render_header("🔄 ML Pipeline",
              "End-to-end machine learning workflow, orchestrated by Airflow")

cfg = get_config()

# ── DAG Diagram ───────────────────────────────────────────────────────────
st.markdown("### Pipeline Architecture")

nodes = {
    "Data\nIngestion":      (0.08, 0.5, "#10b981"),
    "Validation":           (0.22, 0.5, "#10b981"),
    "Feature\nEngineering": (0.38, 0.5, "#0ea5e9"),
    "Model\nTraining":      (0.55, 0.72, "#f59e0b"),
    "Evaluation":           (0.70, 0.72, "#f59e0b"),
    "Deployment":           (0.86, 0.5, "#8b5cf6"),
    "Drift\nDetection":     (0.55, 0.28, "#ef4444"),
    "Monitoring":           (0.70, 0.28, "#ef4444"),
    "Retrain\nTrigger":     (0.86, 0.28, "#ef4444"),
}
edges = [
    ("Data\nIngestion", "Validation"), ("Validation", "Feature\nEngineering"),
    ("Feature\nEngineering", "Model\nTraining"), ("Model\nTraining", "Evaluation"),
    ("Evaluation", "Deployment"), ("Feature\nEngineering", "Drift\nDetection"),
    ("Drift\nDetection", "Monitoring"), ("Monitoring", "Retrain\nTrigger"),
    ("Retrain\nTrigger", "Model\nTraining"),
]

fig = go.Figure()
for src, dst in edges:
    x0, y0, _ = nodes[src]; x1, y1, _ = nodes[dst]
    fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                             line=dict(color="#cbd5e1", width=2),
                             hoverinfo="none", showlegend=False))
for name, (x, y, color) in nodes.items():
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode="markers+text",
        marker=dict(size=56, color=color, line=dict(width=3, color="white")),
        text=[name], textposition="middle center",
        textfont=dict(size=10, color="white", family="Arial Black"),
        hoverinfo="text", hovertext=name.replace("\n", " "), showlegend=False,
    ))
fig.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.1, 0.9]),
                  margin=dict(l=20, r=20, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

# ── Stage details ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Stage Details")
stages = [
    ("📥", "Data Ingestion", "Load raw CSV, validate schema, drop leaky columns "
     "(TWF/HDF/PWF/OSF/RNF/UDI/Product ID), stratified 80/20 split.",
     "src/data_ingestion.py", "train.csv, test.csv", "5–10 s"),
    ("🔧", "Feature Engineering", "Derive temp_diff, power, wear_degree, speed_torque_ratio, "
     "type_encoded. Fit StandardScaler on train only.",
     "src/data_preprocessing.py", "scaler.joblib, drift_baselines.json", "10–20 s"),
    ("🧠", "Model Training", "RandomForest grid search (8 combos). Every run logged to MLflow "
     "(params, metrics, feature importance, model artifact).",
     "src/model_training.py", "best_model.joblib, test_metrics.json", "60–180 s ⚠️ bottleneck"),
    ("📊", "Evaluation", "Evaluate on hold-out test set. Best model registered "
     "in MLflow Model Registry.",
     "MLflow Registry", "Registered model version", "3–8 s"),
    ("🚀", "Deployment", "FastAPI in Docker. Predictions logged to SQLite for "
     "ground-truth feedback matching.",
     "api/main.py", "/predict, /feedback endpoints", "< 1 s (restart)"),
    ("📡", "Monitoring & Drift", "Prometheus scrapes /metrics. Drift via KS-test + PSI. "
     "Alerts on error rate >5%, p95 >200ms, or drift.",
     "Prometheus + Grafana", "Alerts, dashboards, drift reports", "5–10 s"),
    ("🔁", "Retrain Trigger", "When drift detected or feedback accuracy degrades, "
     "authenticated /retrain endpoint triggers the Airflow DAG. "
     "Upload new CSV directly from the Monitoring page.",
     "airflow/dags/ml_pipeline_dag.py", "New model version", "~2–4 min (full pipeline)"),
]
for i, (icon, name, desc, module, outputs, est_time) in enumerate(stages):
    with st.expander(f"{icon} Stage {i+1}: {name}", expanded=i == 0):
        st.markdown(f"**What it does.** {desc}")
        st.markdown(f"**Module.** `{module}`")
        st.markdown(f"**Outputs.** `{outputs}`")
        st.markdown(f"**Estimated time.** `{est_time}`")

st.markdown("---")
st.markdown("### Open external tools")
from frontend.common import _is_reachable
tools_pipeline = [
    ("🔬 MLflow",     cfg.mlflow_url),
    ("🌀 Airflow",    cfg.airflow_url),
    ("📊 Grafana",    cfg.grafana_url),
    ("🔢 Prometheus", cfg.prometheus_url),
]
cols = st.columns(len(tools_pipeline))
for col, (label, url) in zip(cols, tools_pipeline):
    with col:
        if _is_reachable(url):
            st.link_button(label, url, use_container_width=True)
        else:
            st.button(
                f"{label} — offline",
                disabled=True,
                use_container_width=True,
                help=f"{url} is not reachable. Start the service or use Docker Compose.",
            )