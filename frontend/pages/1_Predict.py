"""
Prediction Page — Single, batch predictions, and ground-truth feedback.
"""

import pandas as pd
import plotly.graph_objects as go
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
                    result = client.predict_batch(readings)
            except APIError as e:
                st.error(f"Batch failed — {e}")
            else:
                b1, b2, b3 = st.columns(3)
                b1.metric("Total", result["total"])
                b2.metric("Failures", result["failures_detected"])
                b3.metric("Failure Rate",
                          f"{result['failures_detected']/max(result['total'],1):.1%}")
                preds_df = pd.DataFrame(result["predictions"])
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
