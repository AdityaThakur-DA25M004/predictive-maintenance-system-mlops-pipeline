"""
Prediction Page — Single, batch predictions, and ground-truth feedback.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from frontend.common import APIError, get_client, render_header, render_sidebar, setup_page

setup_page("Predict", icon="🔮")
render_sidebar()
render_header("🔮 Machine Failure Prediction",
              "Enter sensor readings to estimate failure probability")

client = get_client()
tab_single, tab_batch, tab_feedback = st.tabs([
    "🎯 Single Prediction", "📂 Batch (CSV)", "🔁 Submit Ground Truth",
])

# ══ Single Prediction ═════════════════════════════════════════════════════
with tab_single:
    st.markdown("#### Sensor Readings")
    with st.form("predict_form", clear_on_submit=False):
        r1, r2, r3 = st.columns(3)
        with r1:
            air_temp = st.number_input("Air Temperature (K)", 250.0, 350.0, 300.0, 0.1)
        with r2:
            proc_temp = st.number_input("Process Temperature (K)", 250.0, 400.0, 310.0, 0.1)
        with r3:
            rpm = st.number_input("Rotational Speed (RPM)", 0, 5000, 1500, 10)
        r4, r5, r6 = st.columns(3)
        with r4:
            torque = st.number_input("Torque (Nm)", 0.0, 200.0, 40.0, 0.5)
        with r5:
            wear = st.number_input("Tool Wear (min)", 0, 300, 100, 1)
        with r6:
            ptype = st.selectbox("Product Type", ["L", "M", "H"])
        submitted = st.form_submit_button("🔍 Predict", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "air_temperature": air_temp, "process_temperature": proc_temp,
            "rotational_speed": rpm, "torque": torque,
            "tool_wear": wear, "product_type": ptype,
        }
        try:
            with st.spinner("Scoring…"):
                result = client.predict(payload)
        except APIError as e:
            st.error(f"Prediction failed — {e}")
        else:
            st.session_state["last_prediction"] = result
            risk = result["risk_level"]
            prob = result["failure_probability"]

            st.markdown("#### Result")
            actions = {
                "LOW": "Normal operation — continue monitoring.",
                "MEDIUM": "Schedule maintenance within the next shift.",
                "HIGH": "Prioritize inspection before next run.",
                "CRITICAL": "Stop machine immediately, inspect now.",
            }
            icons = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "⚠️", "CRITICAL": "🚨"}
            st.markdown(f"**{icons[risk]} {risk}** — {actions[risk]}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Prediction", "⚠️ FAILURE" if result["prediction"] == 1 else "✅ HEALTHY")
            c2.metric("Failure Probability", f"{prob:.1%}")
            c3.metric("Risk Level", risk)
            c4.metric("Prediction ID", result.get("prediction_id", "—"))

            # Inference time row
            inf_ms = result.get("inference_time_ms")
            if inf_ms is not None:
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Inference Time", f"{inf_ms:.1f} ms",
                           help="Pure model scoring time (excludes API/network overhead)")
                slo_color = "🟢" if inf_ms < 50 else ("🟡" if inf_ms < 200 else "🔴")
                t2.metric("SLO Status (200 ms)", f"{slo_color} {'OK' if inf_ms < 200 else 'SLOW'}")
                t3.metric("Model Version", result.get("model_version", "—"))
                t4.metric("Algorithm", "Random Forest")

            # Gauge
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100,
                number={"suffix": "%", "font": {"size": 42}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#0f172a"},
                    "steps": [
                        {"range": [0, 20], "color": "#dcfce7"},
                        {"range": [20, 50], "color": "#fef3c7"},
                        {"range": [50, 80], "color": "#fed7aa"},
                        {"range": [80, 100], "color": "#fecaca"},
                    ],
                },
            ))
            gauge.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(gauge, use_container_width=True)

            st.info(f"ℹ️ Keep prediction ID **{result.get('prediction_id')}**. "
                    "Submit the actual outcome in the *Submit Ground Truth* tab.")

# ══ Batch Prediction ═════════════════════════════════════════════════════
with tab_batch:
    st.markdown("#### Upload CSV of sensor readings")
    st.caption("Columns: `air_temperature`, `process_temperature`, "
               "`rotational_speed`, `torque`, `tool_wear`, `product_type`")
    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        rename = {"Air temperature [K]": "air_temperature",
                  "Process temperature [K]": "process_temperature",
                  "Rotational speed [rpm]": "rotational_speed",
                  "Torque [Nm]": "torque", "Tool wear [min]": "tool_wear",
                  "Type": "product_type"}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        req = ["air_temperature", "process_temperature", "rotational_speed",
               "torque", "tool_wear", "product_type"]
        missing = [c for c in req if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        elif st.button("🚀 Run Batch Prediction", type="primary"):
            readings = df[req].to_dict(orient="records")
            try:
                with st.spinner(f"Scoring {len(readings)} readings…"):
                    import time as _time
                    t0 = _time.time()
                    result = client.predict_batch(readings)
                    elapsed_ms = (_time.time() - t0) * 1000
            except APIError as e:
                st.error(f"Batch failed — {e}")
            else:
                total = result["total"]
                failures = result["failures_detected"]
                per_row_ms = elapsed_ms / max(total, 1)

                b1, b2, b3, b4, b5 = st.columns(5)
                b1.metric("Total", total)
                b2.metric("Failures", failures)
                b3.metric("Failure Rate", f"{failures/max(total,1):.1%}")
                b4.metric("Total Time", f"{elapsed_ms:.0f} ms")
                b5.metric("Per-row", f"{per_row_ms:.1f} ms")

                # Risk distribution donut
                preds_df = pd.DataFrame(result["predictions"])
                if "risk_level" in preds_df.columns:
                    risk_counts = preds_df["risk_level"].value_counts().reset_index()
                    risk_counts.columns = ["Risk Level", "Count"]
                    risk_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
                    risk_color_map = {
                        "LOW": "#22c55e", "MEDIUM": "#eab308",
                        "HIGH": "#f97316", "CRITICAL": "#ef4444",
                    }
                    risk_counts["Risk Level"] = pd.Categorical(
                        risk_counts["Risk Level"], categories=risk_order, ordered=True
                    )
                    risk_counts = risk_counts.sort_values("Risk Level")
                    fig_donut = px.pie(
                        risk_counts, names="Risk Level", values="Count",
                        hole=0.55, title="Risk Distribution",
                        color="Risk Level",
                        color_discrete_map=risk_color_map,
                    )
                    fig_donut.update_traces(textposition="outside", textinfo="label+percent")
                    fig_donut.update_layout(
                        height=320, margin=dict(l=10, r=10, t=40, b=10),
                        showlegend=False,
                    )

                    dc1, dc2 = st.columns([1, 2])
                    with dc1:
                        st.plotly_chart(fig_donut, use_container_width=True)
                    with dc2:
                        # Inference time distribution if available
                        if "inference_time_ms" in preds_df.columns:
                            inf_times = preds_df["inference_time_ms"].dropna()
                            if not inf_times.empty:
                                st.markdown("**Inference time (ms)**")
                                t1, t2, t3 = st.columns(3)
                                t1.metric("Min", f"{inf_times.min():.1f}")
                                t2.metric("Mean", f"{inf_times.mean():.1f}")
                                t3.metric("Max", f"{inf_times.max():.1f}")

                        st.dataframe(
                            preds_df[["prediction", "failure_probability",
                                      "risk_level", "inference_time_ms",
                                      "prediction_id"]].head(20),
                            use_container_width=True, hide_index=True,
                        )
                else:
                    st.dataframe(preds_df, use_container_width=True, hide_index=True)

                st.download_button("📥 Download CSV",
                                   preds_df.to_csv(index=False).encode(),
                                   "predictions.csv", "text/csv")

# ══ Ground-Truth Feedback ════════════════════════════════════════════════
with tab_feedback:
    st.markdown("#### Report the actual outcome")
    st.caption("Submit ground truth to track live model accuracy.")
    default_id = st.session_state.get("last_prediction", {}).get("prediction_id", 1)
    fc1, fc2 = st.columns(2)
    with fc1:
        pid = st.number_input("Prediction ID", min_value=1, value=max(1, int(default_id or 1)), step=1)
    with fc2:
        actual = st.radio("Actual outcome",
                          [("Healthy (no failure)", 0), ("Failure occurred", 1)],
                          format_func=lambda x: x[0], horizontal=True)
    if st.button("📝 Submit feedback", type="primary"):
        try:
            res = client.submit_feedback(int(pid), int(actual[1]))
        except APIError as e:
            st.error(f"Could not record feedback — {e}")
        else:
            msg = "✅ Correct" if res["correct"] else "❌ Incorrect"
            st.success(f"Feedback recorded. {msg}.")
            if res.get("rolling_accuracy") is not None:
                st.metric("Rolling accuracy", f"{res['rolling_accuracy']:.1%}")

    st.markdown("---")
    try:
        stats = client.feedback_stats()
        if stats["total_feedback"] > 0:
            s1, s2, s3 = st.columns(3)
            s1.metric("Total feedback", stats["total_feedback"])
            s2.metric("Overall accuracy",
                      f"{stats['overall_accuracy']:.1%}" if stats["overall_accuracy"] else "—")
            s3.metric(f"Rolling (last {stats['window']})",
                      f"{stats['rolling_accuracy']:.1%}" if stats["rolling_accuracy"] else "—")
        else:
            st.info("No feedback yet. Submit some to start tracking live accuracy.")
    except APIError:
        st.warning("Cannot fetch feedback stats — is the API running?")