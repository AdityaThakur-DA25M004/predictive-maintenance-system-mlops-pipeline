"""
Predictive Maintenance — Main Dashboard.

Visual upgrade vs previous version:
  * Hero band with embedded live stats (predictions today, last train, drift)
  * Health pills coloured by status (green/amber/red) instead of flat metrics
  * Performance gauges with progress bars + qualitative labels
  * Pipeline timeline showing which stages have run
  * Polished quick-action cards instead of a plain info box
"""
import sys
import os
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from frontend.common import APIError, get_client, render_sidebar, setup_page

setup_page("Dashboard", icon="🏭")
render_sidebar()


# ── Page-specific CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hero with embedded stats */
.pm-hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #6366f1 50%, #0ea5e9 100%);
    color: white;
    padding: 2rem 2.2rem;
    border-radius: 16px;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 30px -10px rgba(30, 58, 138, 0.5);
    position: relative;
    overflow: hidden;
}
.pm-hero::before {
    content: "";
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    pointer-events: none;
}
.pm-hero h1 {
    color: white; margin: 0; font-size: 2.0rem;
    font-weight: 700; letter-spacing: -0.5px;
}
.pm-hero p.tagline {
    color: #e0f2fe; margin: 0.4rem 0 1.4rem 0;
    font-size: 1.0rem;
}
.hero-stats {
    display: flex; gap: 2.5rem; flex-wrap: wrap;
    margin-top: 0.4rem; position: relative; z-index: 1;
}
.hero-stat-label {
    font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 1px; color: #bae6fd; margin: 0;
}
.hero-stat-value {
    font-size: 1.6rem; font-weight: 700;
    color: white; margin: 0.2rem 0 0 0;
}

/* Section headers */
.pm-section {
    font-size: 1.05rem; font-weight: 600;
    color: #1e293b; margin: 1.6rem 0 0.8rem 0;
    display: flex; align-items: center; gap: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #e2e8f0;
}

/* Status pill cards */
.status-card {
    background: white; border-radius: 12px;
    padding: 1.0rem 1.2rem;
    border-left: 4px solid #cbd5e1;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    height: 100%;
}
.status-card.ok    { border-left-color: #10b981; background: linear-gradient(135deg, #ecfdf5 0%, #ffffff 60%); }
.status-card.warn  { border-left-color: #f59e0b; background: linear-gradient(135deg, #fffbeb 0%, #ffffff 60%); }
.status-card.bad   { border-left-color: #ef4444; background: linear-gradient(135deg, #fef2f2 0%, #ffffff 60%); }
.status-card-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
.status-card-value { font-size: 1.4rem; font-weight: 700; color: #0f172a; margin-top: 0.25rem; }
.status-card-sub   { font-size: 0.78rem; color: #475569; margin-top: 0.15rem; }

/* Performance gauge cards */
.gauge-card {
    background: white; border-radius: 12px; padding: 1.1rem 1.3rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border: 1px solid #e2e8f0;
    height: 100%;
}
.gauge-label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.6px; }
.gauge-value { font-size: 2.0rem; font-weight: 800; color: #0f172a; margin: 0.2rem 0 0.5rem 0; line-height: 1; }
.gauge-bar { height: 6px; background: #f1f5f9; border-radius: 999px; overflow: hidden; }
.gauge-bar-fill { height: 100%; border-radius: 999px; transition: width .3s ease; }
.gauge-bar-fill.good { background: linear-gradient(90deg, #10b981 0%, #34d399 100%); }
.gauge-bar-fill.ok   { background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%); }
.gauge-bar-fill.poor { background: linear-gradient(90deg, #ef4444 0%, #f87171 100%); }
.gauge-tag { font-size: 0.7rem; font-weight: 600; margin-top: 0.4rem; display: inline-block;
             padding: 0.15rem 0.55rem; border-radius: 999px; }
.gauge-tag.good { background: #d1fae5; color: #065f46; }
.gauge-tag.ok   { background: #fef3c7; color: #92400e; }
.gauge-tag.poor { background: #fee2e2; color: #991b1b; }

/* Quick-action cards */
.action-card {
    background: white; border-radius: 14px; padding: 1.2rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    text-align: center;
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
    height: 100%;
}
.action-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px -6px rgba(30, 58, 138, 0.2);
    border-color: #6366f1;
}
.action-icon { font-size: 1.8rem; }
.action-title { font-weight: 700; color: #0f172a; margin-top: 0.4rem; font-size: 1.0rem; }
.action-desc  { font-size: 0.8rem; color: #64748b; margin-top: 0.25rem; }

/* Pipeline timeline */
.timeline { display: flex; align-items: center; gap: 0; flex-wrap: wrap; }
.timeline-step {
    flex: 1; min-width: 80px; text-align: center;
    padding: 0.5rem 0.2rem;
}
.timeline-dot {
    width: 32px; height: 32px; border-radius: 50%;
    background: #e2e8f0; color: #94a3b8;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.95rem; font-weight: 700;
    margin-bottom: 0.4rem;
    border: 3px solid transparent;
}
.timeline-dot.done   { background: #10b981; color: white; }
.timeline-dot.active { background: #6366f1; color: white; box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2); }
.timeline-dot.idle   { background: #e2e8f0; color: #94a3b8; }
.timeline-label  { font-size: 0.72rem; color: #475569; font-weight: 600; }
.timeline-line {
    flex: 0.5; min-width: 20px; height: 2px;
    background: #e2e8f0; margin-top: -22px;
}
.timeline-line.done { background: #10b981; }
</style>
""", unsafe_allow_html=True)


# ── Data fetching ──────────────────────────────────────────────────────────
client = get_client()

try:
    health = client.health()
    api_ok = True
except APIError:
    health = {}
    api_ok = False

try:
    info = client.model_info()
    metrics = info.get("test_metrics") or {}
    model_version = info.get("model_version", "—")
except APIError:
    info, metrics, model_version = {}, {}, "—"

try:
    fb = client.feedback_stats()
except APIError:
    fb = {}

try:
    uploads = client.list_uploads()
    upload_list = uploads.get("uploads", [])
except APIError:
    upload_list = []


# ── Hero band with embedded stats ──────────────────────────────────────────
up_s = int(health.get("uptime_seconds", 0))
if up_s >= 3600:
    uptime_str = f"{up_s // 3600}h {(up_s % 3600) // 60}m"
elif up_s >= 60:
    uptime_str = f"{up_s // 60}m {up_s % 60}s"
else:
    uptime_str = f"{up_s}s"

f1_disp = f"{metrics.get('f1_score', 0):.3f}" if metrics else "—"
total_preds = fb.get("total_feedback", 0)
total_uploads = len(upload_list)

st.markdown(f"""
<div class="pm-hero">
  <h1>🏭 Predictive Maintenance Dashboard</h1>
  <p class="tagline">Real-time machine health monitoring &amp; failure prediction</p>
  <div class="hero-stats">
    <div>
      <p class="hero-stat-label">Model F1</p>
      <p class="hero-stat-value">{f1_disp}</p>
    </div>
    <div>
      <p class="hero-stat-label">Model Version</p>
      <p class="hero-stat-value">v{model_version}</p>
    </div>
    <div>
      <p class="hero-stat-label">Datasets Uploaded</p>
      <p class="hero-stat-value">{total_uploads}</p>
    </div>
    <div>
      <p class="hero-stat-label">Feedback Logged</p>
      <p class="hero-stat-value">{total_preds}</p>
    </div>
    <div>
      <p class="hero-stat-label">Uptime</p>
      <p class="hero-stat-value">{uptime_str}</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── System health: colour-coded status cards ──────────────────────────────
st.markdown('<div class="pm-section">📡 System Health</div>', unsafe_allow_html=True)


def _status_card(label: str, ok: bool, value_ok: str, value_bad: str, sub: str = ""):
    cls = "ok" if ok else "bad"
    val = value_ok if ok else value_bad
    return f"""
    <div class="status-card {cls}">
      <div class="status-card-label">{label}</div>
      <div class="status-card-value">{val}</div>
      <div class="status-card-sub">{sub}</div>
    </div>
    """


sc1, sc2, sc3, sc4 = st.columns(4)
with sc1:
    st.markdown(_status_card(
        "API Status", api_ok, "🟢 Online", "🔴 Offline",
        "FastAPI reachable" if api_ok else "Cannot reach backend"
    ), unsafe_allow_html=True)
with sc2:
    model_ok = bool(health.get("model_loaded"))
    st.markdown(_status_card(
        "Model", model_ok, "🟢 Loaded", "🔴 Not Loaded",
        f"v{model_version}" if model_ok else "Run training pipeline"
    ), unsafe_allow_html=True)
with sc3:
    sc_ok = bool(health.get("scaler_loaded"))
    st.markdown(_status_card(
        "Scaler", sc_ok, "🟢 Loaded", "🔴 Not Loaded",
        "StandardScaler ready" if sc_ok else "Run preprocessing"
    ), unsafe_allow_html=True)
with sc4:
    st.markdown(f"""
    <div class="status-card ok">
      <div class="status-card-label">Uptime</div>
      <div class="status-card-value">⏱️ {uptime_str}</div>
      <div class="status-card-sub">Since last container restart</div>
    </div>
    """, unsafe_allow_html=True)


# ── Model performance gauges ───────────────────────────────────────────────
st.markdown('<div class="pm-section">📈 Model Performance</div>', unsafe_allow_html=True)


def _qual(value: float) -> tuple[str, str]:
    """Return (css-class, label) based on metric value."""
    if value >= 0.85:
        return "good", "Excellent"
    if value >= 0.70:
        return "ok", "Acceptable"
    return "poor", "Needs work"


def _gauge_card(label: str, value: float, fmt: str = "{:.3f}"):
    cls, tag_label = _qual(value)
    pct = max(0, min(100, value * 100))
    return f"""
    <div class="gauge-card">
      <div class="gauge-label">{label}</div>
      <div class="gauge-value">{fmt.format(value)}</div>
      <div class="gauge-bar">
        <div class="gauge-bar-fill {cls}" style="width: {pct}%;"></div>
      </div>
      <span class="gauge-tag {cls}">{tag_label}</span>
    </div>
    """


if metrics:
    g1, g2, g3, g4, g5 = st.columns(5)
    with g1: st.markdown(_gauge_card("F1 Score", metrics.get("f1_score", 0)), unsafe_allow_html=True)
    with g2: st.markdown(_gauge_card("ROC-AUC",  metrics.get("roc_auc", 0)),  unsafe_allow_html=True)
    with g3: st.markdown(_gauge_card("Accuracy", metrics.get("accuracy", 0)), unsafe_allow_html=True)
    with g4: st.markdown(_gauge_card("Precision", metrics.get("precision", 0)), unsafe_allow_html=True)
    with g5: st.markdown(_gauge_card("Recall",   metrics.get("recall", 0)),   unsafe_allow_html=True)
else:
    st.info("📦 Model metrics not yet available. Run the training pipeline first.")


# ── Live feedback (only show if feedback exists) ──────────────────────────
if fb.get("total_feedback", 0) > 0:
    st.markdown('<div class="pm-section">🔁 Live Feedback Accuracy</div>', unsafe_allow_html=True)
    f1col, f2col, f3col = st.columns(3)
    with f1col:
        st.markdown(f"""
        <div class="status-card ok">
          <div class="status-card-label">Feedback Received</div>
          <div class="status-card-value">{fb['total_feedback']}</div>
          <div class="status-card-sub">Ground-truth labels logged</div>
        </div>""", unsafe_allow_html=True)
    with f2col:
        ov = fb.get("overall_accuracy")
        st.markdown(_gauge_card(
            "Overall Accuracy", ov or 0, fmt="{:.1%}"
        ), unsafe_allow_html=True)
    with f3col:
        rl = fb.get("rolling_accuracy")
        st.markdown(_gauge_card(
            f"Rolling (last {fb.get('window', '?')})", rl or 0, fmt="{:.1%}"
        ), unsafe_allow_html=True)


# ── Pipeline status timeline ───────────────────────────────────────────────
st.markdown('<div class="pm-section">🔄 Pipeline Status</div>', unsafe_allow_html=True)

# Status logic: green if model loaded (means full pipeline ran at least once),
# amber if only API is up, grey otherwise.
have_model = bool(health.get("model_loaded"))
have_uploads = total_uploads > 0
stages = [
    ("📥", "Ingest",     "done" if have_model or have_uploads else "idle"),
    ("🔍", "Drift Check","done" if have_model else "idle"),
    ("⚙️", "Preprocess", "done" if have_model else "idle"),
    ("🎯", "Train",      "done" if have_model else "idle"),
    ("🚀", "Deploy",     "active" if have_model else "idle"),
]

timeline_html = '<div class="timeline">'
for i, (icon, label, status) in enumerate(stages):
    if i > 0:
        line_status = "done" if status in ("done", "active") and stages[i-1][2] == "done" else ""
        timeline_html += f'<div class="timeline-line {line_status}"></div>'
    timeline_html += f"""
    <div class="timeline-step">
      <div class="timeline-dot {status}">{icon}</div>
      <div class="timeline-label">{label}</div>
    </div>"""
timeline_html += '</div>'
st.markdown(timeline_html, unsafe_allow_html=True)

if upload_list:
    latest_ts = max(u["uploaded_at"] for u in upload_list)
    latest_str = datetime.datetime.fromtimestamp(latest_ts).strftime("%Y-%m-%d %H:%M")
    st.caption(f"📂 Latest dataset upload: **{latest_str}** · {total_uploads} total")


# ── System overview (kept, slightly polished) ──────────────────────────────
st.markdown('<div class="pm-section">🎯 System Overview</div>', unsafe_allow_html=True)
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
**Stack:**
- 🔧 **Backend:** FastAPI + Prometheus instrumentation
- 🧠 **ML:** Random Forest + MLflow tracking & registry
- 🔄 **Pipeline:** Airflow DAG with drift-triggered retraining
- 📊 **Monitoring:** Prometheus + Grafana + alert rules
- 📦 **Versioning:** DVC + DagsHub
- 🐳 **Deploy:** Docker Compose (8 containers)
    """)


# ── Quick action cards ─────────────────────────────────────────────────────
st.markdown('<div class="pm-section">🚀 Quick Actions</div>', unsafe_allow_html=True)

actions = [
    ("🔮", "Predict",     "Submit a sensor reading & get a failure probability"),
    ("🔄", "Pipeline",    "Trigger retraining or upload a new dataset"),
    ("📊", "Monitoring",  "Inspect drift, model quality, and alerts"),
    ("📘", "User Manual", "How to use the system end-to-end"),
]
ac1, ac2, ac3, ac4 = st.columns(4)
for col, (icon, title, desc) in zip([ac1, ac2, ac3, ac4], actions):
    with col:
        st.markdown(f"""
        <div class="action-card">
          <div class="action-icon">{icon}</div>
          <div class="action-title">{title}</div>
          <div class="action-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.caption(
    "Use the sidebar to navigate between pages. "
    "External tools (MLflow, Airflow, Grafana, Prometheus, API docs) are linked there too."
)