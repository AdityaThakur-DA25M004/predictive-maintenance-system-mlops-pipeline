"""
Monitoring Page — Model quality, drift detection, retrain triggers.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.common import (
    APIError, get_client, get_config, render_header, render_sidebar, setup_page,
)

setup_page("Monitoring", icon="📊")
render_sidebar()
render_header("📊 System Monitoring & Drift Detection",
              "Model quality, data drift, and retraining controls")

client = get_client()
cfg = get_config()

tab_model, tab_drift, tab_retrain = st.tabs([
    "📈 Model Quality", "📉 Drift Detection", "🔁 Retrain",
])

# ══ Model Quality ════════════════════════════════════════════════════════
with tab_model:
    try:
        info = client.model_info()
        metrics = info.get("test_metrics") or {}
    except APIError as e:
        st.warning(f"Cannot load model info — {e}")
        metrics = {}

    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        c2.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
        c3.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        c4.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        c5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")

        st.markdown("#### Confusion Matrix")
        cm = np.array([
            [metrics.get("true_negatives", 0), metrics.get("false_positives", 0)],
            [metrics.get("false_negatives", 0), metrics.get("true_positives", 0)],
        ])
        fig = px.imshow(cm, text_auto=True,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["Healthy", "Failure"], y=["Healthy", "Failure"],
                        color_continuous_scale="Blues")
        fig.update_layout(height=360, margin=dict(l=40, r=40, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No test metrics available yet. Run the training pipeline.")

    st.markdown("---")
    st.markdown("#### 🔁 Live Feedback Accuracy")
    try:
        fb = client.feedback_stats()
        if fb["total_feedback"] == 0:
            st.info("No ground-truth feedback submitted yet.")
        else:
            f1, f2, f3 = st.columns(3)
            f1.metric("Total feedback", fb["total_feedback"])
            ov = fb.get("overall_accuracy")
            rl = fb.get("rolling_accuracy")
            f2.metric("Overall", f"{ov:.1%}" if ov is not None else "—")
            f3.metric(f"Rolling (last {fb['window']})", f"{rl:.1%}" if rl is not None else "—")
    except APIError:
        st.warning("Feedback stats unavailable.")

# ══ Drift Detection ══════════════════════════════════════════════════════
with tab_drift:
    st.markdown("#### Check for distribution drift against training baselines")
    mode = st.radio("Data source", ["Simulated", "Upload CSV"], horizontal=True)
    readings = None

    if mode == "Simulated":
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider("Samples", 50, 1000, 200)
        with col2:
            scenario = st.selectbox("Scenario", [
                "No drift (baseline)", "Temperature drift (+3σ)",
                "Torque drift (+1.5σ)", "Full drift (all shifted)",
            ])
        if st.button("🎲 Generate & check", type="primary"):
            rng = np.random.default_rng(42)
            readings = []
            for _ in range(n_samples):
                if scenario.startswith("No drift"):
                    at, tq, sp = rng.normal(300, 2), rng.normal(40, 10), int(rng.normal(1500, 200))
                elif scenario.startswith("Temperature"):
                    at, tq, sp = rng.normal(310, 3), rng.normal(40, 10), int(rng.normal(1500, 200))
                elif scenario.startswith("Torque"):
                    at, tq, sp = rng.normal(300, 2), rng.normal(60, 15), int(rng.normal(1500, 200))
                else:
                    at, tq, sp = rng.normal(310, 4), rng.normal(60, 20), int(rng.normal(1200, 300))
                readings.append({
                    "air_temperature": float(max(at, 260)),
                    "process_temperature": float(max(at + rng.normal(10, 1), 260)),
                    "rotational_speed": max(sp, 100),
                    "torque": float(max(tq, 1)),
                    "tool_wear": int(rng.integers(0, 250)),
                    "product_type": rng.choice(["H", "M", "L"]),
                })
    else:
        up = st.file_uploader("Upload CSV", type=["csv"], key="drift_upload")
        if up and st.button("🔍 Check uploaded data", type="primary"):
            df = pd.read_csv(up)
            rename = {"Air temperature [K]": "air_temperature",
                      "Process temperature [K]": "process_temperature",
                      "Rotational speed [rpm]": "rotational_speed",
                      "Torque [Nm]": "torque", "Tool wear [min]": "tool_wear",
                      "Type": "product_type"}
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
            req = ["air_temperature", "process_temperature", "rotational_speed",
                   "torque", "tool_wear", "product_type"]
            if any(c not in df.columns for c in req):
                st.error(f"Missing: {[c for c in req if c not in df.columns]}")
            else:
                readings = df[req].to_dict(orient="records")

    if readings:
        try:
            with st.spinner("Checking…"):
                drift = client.drift_check(readings)
        except APIError as e:
            st.error(f"Drift check failed — {e}")
        else:
            if drift["overall_drift"]:
                st.error(f"⚠️ **Drift detected** in {drift['n_drifted']} / "
                         f"{drift['total_features_checked']} features: "
                         f"{', '.join(drift['drifted_features'])}")
            else:
                st.success(f"✅ No drift. Checked {drift['total_features_checked']} features.")
            rows = []
            for feat, d in drift["features"].items():
                if isinstance(d, dict) and "ks_p_value" in d:
                    rows.append({
                        "Feature": feat,
                        "KS p-value": f"{d['ks_p_value']:.4f}",
                        "PSI": f"{d['psi']:.4f}",
                        "Mean shift (σ)": f"{d.get('mean_shift_std', 0):.2f}",
                        "Drift?": "🔴 YES" if d["drift_detected"] else "🟢 NO",
                    })
            if rows:
                st.markdown("#### Feature-level results")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══ Retrain ══════════════════════════════════════════════════════════════
with tab_retrain:
    st.markdown("#### Manual retrain trigger")
    st.caption("Protected by API key (`RETRAIN_API_KEY` env var).")
    r1, r2 = st.columns([2, 3])
    with r1:
        reason = st.selectbox("Reason", ["manual", "drift_detected", "performance_degradation", "scheduled"])
    with r2:
        api_key = st.text_input("API key", type="password")
    if st.button("🔁 Trigger retrain", type="primary", disabled=not api_key):
        try:
            res = client.trigger_retrain(reason, api_key)
        except APIError as e:
            if e.status_code == 401:
                st.error("❌ Unauthorized — check the API key.")
            else:
                st.error(f"Trigger failed — {e}")
        else:
            st.success(f"✅ {res['message']}")

    st.markdown("---")
    st.markdown("#### Live dashboards")
    st.link_button("📊 Open Grafana", cfg.grafana_url, use_container_width=True)
    st.link_button("🔢 Open Prometheus", cfg.prometheus_url, use_container_width=True)
