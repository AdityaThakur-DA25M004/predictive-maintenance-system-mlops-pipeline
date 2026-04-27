"""
Monitoring Page — Model quality, drift detection, retrain triggers.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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

tab_model, tab_drift, tab_retrain, tab_alerts = st.tabs([
    "📈 Model Quality", "📉 Drift Detection", "🔁 Retrain", "🚨 Alerts",
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

        # ── Feature importance ─────────────────────────────────────────
        st.markdown("#### 🌲 Feature Importance")
        try:
            fi_data = client.feature_importance()
            fi = fi_data.get("feature_importance", {})
            if fi:
                fi_df = pd.DataFrame(
                    {"Feature": list(fi.keys()), "Importance": list(fi.values())}
                ).sort_values("Importance")
                fig_fi = px.bar(
                    fi_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="Blues",
                    text=fi_df["Importance"].map(lambda v: f"{v:.4f}"),
                )
                fig_fi.update_traces(textposition="outside")
                fig_fi.update_layout(
                    height=380, margin=dict(l=10, r=60, t=10, b=10),
                    coloraxis_showscale=False,
                    yaxis=dict(tickfont=dict(size=12)),
                )
                st.plotly_chart(fig_fi, use_container_width=True)
                top = fi_data.get("top_feature", "—")
                st.caption(f"Most predictive feature: **{top}**")
        except APIError:
            st.info("Feature importance not available — model may not be loaded.")

        # ── Classification Report ──────────────────────────────────────
        st.markdown("#### 📋 Classification Report")
        try:
            report = info.get("classification_report") or {}
            if report:
                cls_rows = []
                for label, label_name in [("0", "Healthy"), ("1", "Failure")]:
                    if label in report:
                        r = report[label]
                        cls_rows.append({
                            "Class": label_name,
                            "Precision": f"{r.get('precision', 0):.3f}",
                            "Recall": f"{r.get('recall', 0):.3f}",
                            "F1-Score": f"{r.get('f1-score', 0):.3f}",
                            "Support": int(r.get("support", 0)),
                        })
                if "macro avg" in report:
                    m = report["macro avg"]
                    cls_rows.append({
                        "Class": "Macro avg",
                        "Precision": f"{m.get('precision', 0):.3f}",
                        "Recall": f"{m.get('recall', 0):.3f}",
                        "F1-Score": f"{m.get('f1-score', 0):.3f}",
                        "Support": int(m.get("support", 0)),
                    })
                if cls_rows:
                    st.dataframe(
                        pd.DataFrame(cls_rows), use_container_width=True, hide_index=True
                    )
        except Exception:
            pass
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
            n_samples = st.slider(
                "Samples", min_value=5, max_value=1000, value=200,
                help="Minimum 3 samples required for statistical testing. "
                     "Use 50+ for reliable results.",
            )
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
    st.markdown("#### 🔁 Retrain Pipeline")

    # ── Pipeline component timing ──────────────────────────────────────
    with st.expander("⏱️ Estimated Pipeline Component Times", expanded=False):
        st.caption("Based on AI4I 2020 dataset (~10 k rows). Times scale with dataset size.")
        timing_data = {
            "Stage": [
                "📥 Data Ingestion",
                "🔧 Feature Engineering & Scaling",
                "🧠 Model Training (grid search, 8 combos)",
                "📡 Drift Detection",
                "🚀 Model Registration (MLflow)",
                "🔄 Total Pipeline",
            ],
            "Estimated Time": ["5–10 s", "10–20 s", "60–180 s", "5–10 s", "3–8 s", "1.5–4 min"],
            "Bottleneck?": ["No", "No", "✅ Yes", "No", "No", "—"],
            "Parallelisable?": ["No", "No", "Yes (Ray/Dask)", "No", "No", "—"],
        }
        st.dataframe(pd.DataFrame(timing_data), use_container_width=True, hide_index=True)

    st.divider()

    # ── Upload → Store → Retrain ───────────────────────────────────────
    st.markdown("##### 📂 Upload New Dataset → Store → Retrain")
    st.caption(
        "Upload a raw CSV matching the AI4I schema. It will be validated, stored on the server, "
        "and marked ready for the next Airflow run. Requires API key."
    )

    with st.expander("📋 Required CSV columns", expanded=False):
        st.code(
            "Type, Air temperature [K], Process temperature [K],\n"
            "Rotational speed [rpm], Torque [Nm], Tool wear [min], Machine failure",
            language="text",
        )
        st.caption("Optional: UDI, Product ID, TWF, HDF, PWF, OSF, RNF (leaky columns are dropped automatically)")

    up_col1, up_col2 = st.columns([3, 2])
    with up_col1:
        upload_file = st.file_uploader(
            "Select CSV file", type=["csv"], key="retrain_upload",
            help="Raw sensor data CSV — will be validated before storage",
        )
    with up_col2:
        upload_reason = st.selectbox(
            "Retrain reason",
            ["csv_upload", "drift_detected", "performance_degradation", "scheduled", "manual"],
            key="upload_reason",
        )
        upload_key = st.text_input("API key", type="password", key="upload_api_key")

    if upload_file is not None:
        try:
            preview_df = pd.read_csv(upload_file)
            upload_file.seek(0)  # reset for actual upload
            st.markdown(f"**Preview** — {len(preview_df):,} rows × {len(preview_df.columns)} columns")
            st.dataframe(preview_df.head(5), use_container_width=True, hide_index=True)

            # Quick schema check in UI before sending
            required_cols = [
                "Type", "Air temperature [K]", "Process temperature [K]",
                "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure",
            ]
            missing_cols = [c for c in required_cols if c not in preview_df.columns]
            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
            else:
                st.success(f"✅ Schema valid — {len(preview_df):,} rows ready to upload")

                col_btn1, col_btn2 = st.columns([1, 3])
                with col_btn1:
                    do_upload = st.button(
                        "🚀 Upload & Store",
                        type="primary",
                        disabled=not upload_key,
                        use_container_width=True,
                    )
                with col_btn2:
                    if not upload_key:
                        st.warning("⚠️ Enter API key to enable upload")

                if do_upload and upload_key:
                    file_bytes = upload_file.read()
                    try:
                        with st.spinner(f"Uploading {len(preview_df):,} rows to server…"):
                            res = client.retrain_with_upload(
                                file_bytes, upload_file.name, upload_reason, upload_key
                            )
                        st.success(f"✅ {res['message']}")
                        uc1, uc2, uc3 = st.columns(3)
                        uc1.metric("Rows stored", f"{res['rows']:,}")
                        uc2.metric("Filename", res["filename"])
                        uc3.metric("Status", res["status"])
                        st.info(
                            "**Next step:** Go to [Airflow UI](%s) → trigger DAG "
                            "`predictive_maintenance_pipeline` to train on this dataset." % cfg.airflow_url
                        )
                    except APIError as e:
                        if e.status_code == 401:
                            st.error("❌ Unauthorized — check the API key.")
                        else:
                            st.error(f"Upload failed — {e}")
        except Exception as e:
            st.error(f"Could not read file: {e}")

    st.divider()

    # ── Email alert status (so user knows whether retrain emails will fire) ──
    st.markdown("##### ✉️ Email Alert Configuration")
    import requests as _alerts_requests
    email_sent_count = 0
    log_only_count = 0
    try:
        r_emstat = _alerts_requests.get(
            f"{cfg.prometheus_internal_url}/api/v1/query",
            params={"query": "sum(alert_notifications_total) by (channel)"},
            timeout=3,
        )
        if r_emstat.ok:
            for item in r_emstat.json().get("data", {}).get("result", []):
                ch = item["metric"].get("channel", "unknown")
                val = float(item["value"][1])
                if ch == "email":
                    email_sent_count = int(val)
                elif ch == "log":
                    log_only_count = int(val)
    except Exception:
        pass

    if email_sent_count > 0:
        st.success(
            f"✅ Email alerts are working — **{email_sent_count}** alert email(s) sent so far. "
            f"({log_only_count} additional alerts logged but not emailed.)"
        )
    elif log_only_count > 0:
        st.warning(
            f"⚠️ {log_only_count} alert(s) have been generated but **none were sent by email**. "
            "Either SMTP isn't configured or sending is failing. "
            "Set `ALERT_EMAIL_ENABLED=true`, `ALERT_SMTP_USER`, `ALERT_SMTP_PASSWORD`, and `ALERT_RECIPIENT` "
            "in your `.env` file (use a Gmail App Password — not your normal Gmail password). "
            "Alerts are still recorded in Prometheus and visible in the **Alerts** tab."
        )
    else:
        st.info(
            "ℹ️ No alerts have been triggered yet. When drift is detected, retrain is triggered, or training "
            "completes, alerts will be dispatched via email (if SMTP is configured) and tracked in Prometheus."
        )

    st.divider()

    # ── Manual retrain trigger (key only) ─────────────────────────────
    st.markdown("##### ⚡ Manual Retrain Trigger (no file upload)")
    r1, r2 = st.columns([2, 3])
    with r1:
        reason = st.selectbox(
            "Reason",
            ["manual", "drift_detected", "performance_degradation", "scheduled"],
            key="manual_reason",
        )
    with r2:
        api_key = st.text_input("API key", type="password", key="manual_api_key")
    if st.button("🔁 Trigger retrain", type="primary", disabled=not api_key, key="manual_trigger"):
        try:
            res = client.trigger_retrain(reason, api_key)
        except APIError as e:
            if e.status_code == 401:
                st.error("❌ Unauthorized — check the API key.")
            else:
                st.error(f"Trigger failed — {e}")
        else:
            st.success(f"✅ {res['message']}")

    st.divider()

    # ── Upload history ─────────────────────────────────────────────────
    st.markdown("##### 📁 Upload History")
    try:
        history = client.list_uploads()
        uploads = history.get("uploads", [])
        if not uploads:
            st.info("No CSV files uploaded yet.")
        else:
            import datetime
            hist_rows = []
            for u in uploads:
                ts = datetime.datetime.fromtimestamp(u["uploaded_at"]).strftime("%Y-%m-%d %H:%M:%S")
                hist_rows.append({
                    "Filename": u["filename"],
                    "Rows": f"{u['rows']:,}" if u.get("rows") is not None else "—",
                    "Size": f"{u['size_bytes'] / 1024:.1f} KB",
                    "Uploaded At": ts,
                })
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)
    except APIError:
        st.warning("Could not fetch upload history — is the API running?")

    st.markdown("---")
    st.markdown("#### Live dashboards")
    from frontend.common import _is_reachable
    mon_col1, mon_col2 = st.columns(2)
    with mon_col1:
        # Probe via Docker DNS, but link the user's browser to the host port
        if _is_reachable(cfg.grafana_internal_url):
            st.link_button("📊 Open Grafana", cfg.grafana_url, use_container_width=True)
        else:
            st.button("📊 Grafana — offline", disabled=True, use_container_width=True,
                      help=f"{cfg.grafana_internal_url} not reachable. Use Docker Compose to start Grafana.")
    with mon_col2:
        if _is_reachable(cfg.prometheus_internal_url):
            st.link_button("🔢 Open Prometheus", cfg.prometheus_url, use_container_width=True)
        else:
            st.button("🔢 Prometheus — offline", disabled=True, use_container_width=True,
                      help=f"{cfg.prometheus_internal_url} not reachable. Use Docker Compose to start Prometheus.")

# ══ Alerts ════════════════════════════════════════════════════════════════
with tab_alerts:
    st.markdown("#### 🚨 System Alerts")
    st.caption(
        "Live view of Prometheus alert rules and dispatched notifications. "
        "Drift, model quality, API health, and training lifecycle events are tracked here."
    )

    import requests as _r
    import datetime as _dt

    def _prom_query(expr: str) -> list[dict]:
        try:
            resp = _r.get(
                f"{cfg.prometheus_internal_url}/api/v1/query",
                params={"query": expr},
                timeout=5,
            )
            resp.raise_for_status()
            return resp.json().get("data", {}).get("result", [])
        except Exception as e:
            st.warning(f"Prometheus query failed: {e}")
            return []

    def _prom_alerts() -> list[dict]:
        """Fetch live alert state from Prometheus /api/v1/alerts."""
        try:
            resp = _r.get(
                f"{cfg.prometheus_internal_url}/api/v1/alerts",
                timeout=5,
            )
            resp.raise_for_status()
            return resp.json().get("data", {}).get("alerts", [])
        except Exception:
            return []

    # ── Headline counters ─────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    drift_now = _prom_query("drift_detected")
    drift_now_val = float(drift_now[0]["value"][1]) if drift_now else 0
    c1.metric("Data Drift", "🟥 ACTIVE" if drift_now_val > 0 else "🟢 None")

    retrain_total = _prom_query("sum(retrain_triggers_total)")
    retrain_val = float(retrain_total[0]["value"][1]) if retrain_total else 0
    c2.metric("Retrain Triggers (all-time)", int(retrain_val))

    email_total = _prom_query('sum(alert_notifications_total{channel="email"})')
    email_val = float(email_total[0]["value"][1]) if email_total else 0
    c3.metric("Alert Emails Sent", int(email_val))

    log_total = _prom_query('sum(alert_notifications_total{channel="log"})')
    log_val = float(log_total[0]["value"][1]) if log_total else 0
    c4.metric("Alerts Log-Only", int(log_val),
              help="Alerts generated but not emailed (SMTP unconfigured or failed)")

    st.markdown("---")

    # ── Live firing alerts (from /api/v1/alerts) ──────────────────────
    st.markdown("##### 🔥 Currently firing alerts")
    alerts = _prom_alerts()
    firing = [a for a in alerts if a.get("state") == "firing"]
    pending = [a for a in alerts if a.get("state") == "pending"]

    if firing:
        rows = []
        for a in firing:
            labels = a.get("labels", {})
            ann = a.get("annotations", {})
            rows.append({
                "Alert": labels.get("alertname", "—"),
                "Severity": labels.get("severity", "—"),
                "Category": labels.get("category", "—"),
                "Summary": ann.get("summary", ""),
                "Description": ann.get("description", ""),
                "Active Since": a.get("activeAt", "")[:19].replace("T", " "),
            })
        st.error(f"**{len(firing)} alert(s) currently firing**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    elif pending:
        st.warning(f"⏳ {len(pending)} alert(s) pending (waiting for `for:` duration to elapse)")
    else:
        st.success("✅ No alerts firing.")

    if pending and firing:
        with st.expander(f"⏳ {len(pending)} pending alert(s)", expanded=False):
            for a in pending:
                st.write(f"- **{a['labels'].get('alertname', '—')}** — "
                         f"{a.get('annotations', {}).get('summary', '')}")

    st.markdown("---")

    # ── Alert dispatch breakdown ──────────────────────────────────────
    st.markdown("##### 📬 Alert dispatches by type and channel (last 24h)")
    dispatched = _prom_query(
        "sum(increase(alert_notifications_total[24h])) by (alert_type, channel)"
    )
    if dispatched:
        rows = []
        for r in dispatched:
            metric = r.get("metric", {})
            rows.append({
                "Alert Type": metric.get("alert_type", "unknown"),
                "Channel": metric.get("channel", "unknown"),
                "Count (24h)": float(r["value"][1]),
            })
        df_disp = pd.DataFrame(rows).sort_values(
            ["Alert Type", "Channel"], ascending=[True, False]
        )
        st.dataframe(df_disp, use_container_width=True, hide_index=True)
    else:
        st.info("No alert dispatches in the last 24 hours.")

    st.markdown("---")

    # ── Defined alert rules (helpful for users new to the system) ────
    st.markdown("##### 📋 Configured alert rules")
    with st.expander("View alert rule definitions", expanded=False):
        st.caption("These rules are defined in `monitoring/prometheus/alert_rules.yml`.")
        rules_summary = pd.DataFrame([
            {"Rule": "APIDown",                     "Threshold": "scrape failure for 1m",     "Severity": "critical"},
            {"Rule": "ModelNotLoaded",              "Threshold": "F1 == 0 for 5m",            "Severity": "critical"},
            {"Rule": "HighErrorRate",               "Threshold": "> 5% errors for 2m",        "Severity": "critical"},
            {"Rule": "ElevatedErrorRate",           "Threshold": "> 2% errors for 5m",        "Severity": "warning"},
            {"Rule": "HighLatencyP95",              "Threshold": "P95 > 200ms for 3m",        "Severity": "warning"},
            {"Rule": "VeryHighLatencyP99",          "Threshold": "P99 > 500ms for 5m",        "Severity": "critical"},
            {"Rule": "DataDriftDetected",           "Threshold": "drift_detected==1 for 5m",  "Severity": "warning"},
            {"Rule": "MultipleFeaturesDrifted",     "Threshold": "≥ 3 features for 2m",       "Severity": "critical"},
            {"Rule": "ModelF1Degraded",             "Threshold": "F1 < 0.5 for 10m",          "Severity": "warning"},
            {"Rule": "FeedbackAccuracyDegraded",    "Threshold": "rolling acc < 70% for 15m", "Severity": "warning"},
            {"Rule": "RetrainStorm",                "Threshold": "> 5 retrains in 15m",       "Severity": "warning"},
            {"Rule": "ModelStale",                  "Threshold": "> 7 days since training",   "Severity": "warning"},
            {"Rule": "UploadSuccessRate",           "Threshold": "> 50% upload failures",     "Severity": "warning"},
            {"Rule": "TrainingDataSourceDefault",   "Threshold": "default dataset used",      "Severity": "info"},
            {"Rule": "AlertEmailDispatchFailing",   "Threshold": "log alerts but no email",   "Severity": "warning"},
        ])
        st.dataframe(rules_summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(
        f"💡 For richer time-series and history, use [Grafana]({cfg.grafana_url}) "
        f"or [Prometheus]({cfg.prometheus_url}/alerts) directly."
    )