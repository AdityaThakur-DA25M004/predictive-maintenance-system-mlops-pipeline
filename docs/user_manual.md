# User Manual — Predictive Maintenance System

> **Who this is for.** Maintenance operators, supervisors, and anyone who
> needs to use the system day-to-day without writing code. If you can use
> a web browser, you can use this system.

---

## 1. What this system does

The Predictive Maintenance System watches industrial equipment using
sensor readings (temperature, speed, torque, tool wear, product type)
and predicts whether a machine is likely to fail soon. Early warning
means you can schedule maintenance before something breaks rather than
after — which saves money, time, and avoids safety risks.

Behind the scenes, the system was trained on a real-world dataset of
10,000 industrial machine records covering five common failure types.
You don't need to know any of the technical details to use it. Open the
dashboard, enter a few numbers, and read the result.

---

## 2. Before you start

You need:

- A modern web browser (Chrome, Firefox, Edge, Safari).
- Network access to the system's address. Your administrator will give
  you this. It usually looks like `http://<your-server>:8501`.

That's it. Nothing to install.

---

## 3. The dashboard

Open the dashboard URL in your browser. You'll see five things in the
sidebar:

| Page | What it does |
|---|---|
| **app** | Overview dashboard — system health, model quality, recent activity |
| **Predict** | Submit a sensor reading and get a failure prediction |
| **Pipeline** | Upload a new dataset, retrain the model, roll back to an older one |
| **Monitoring** | Check for drift in your data, view model quality, see active alerts |
| **User Manual** | A built-in copy of this document |

The first thing you'll see on the **app** page is a coloured banner
showing the model's current quality (F1 score), version number, and
how many predictions have been made. Below it are status indicators
("Online", "Loaded") confirming everything is healthy. If you see red
or "Offline" anywhere, contact your administrator.

---

## 4. Making a single prediction

This is the most common task. Use it whenever you want a quick check
on a specific machine.

1. Click **Predict** in the sidebar.
2. Enter the six sensor readings:

| Field | What it means | Typical range |
|---|---|---|
| Air Temperature (K) | Ambient air temperature | 295–305 K |
| Process Temperature (K) | Machine process temperature | 305–315 K |
| Rotational Speed (RPM) | Spindle speed | 1,000–2,500 RPM |
| Torque (Nm) | Applied torque | 20–80 Nm |
| Tool Wear (min) | Cumulative tool usage time | 0–250 min |
| Product Type | Quality grade — H, M, or L | — |

3. Click **🔍 Predict**.
4. Read the result. You'll see four things:

   - **Prediction** — either "No failure expected" or "Failure likely".
   - **Failure probability** — a number from 0% to 100%.
   - **Risk level** — colour-coded interpretation (see §6 below).
   - **Prediction ID** — a unique number. Note it down if you plan to
     submit feedback later (see §7).

---

## 5. Predicting many readings at once (batch)

If you have a CSV file with many sensor readings:

1. Click **Predict** in the sidebar, then switch to the **Batch
   Prediction** tab.
2. Click **Browse files** and select your CSV. The file must have these
   exact column names:
   `air_temperature, process_temperature, rotational_speed, torque,
   tool_wear, product_type`.
3. Click **🚀 Run Batch Prediction**.
4. The results appear in a table — one row per input, with the
   prediction and probability for each.
5. You can download the results as a CSV if you want to share them.

---

## 6. Understanding risk levels

Every prediction comes with a colour-coded risk level. Use this as your
guide to action:

| Risk Level | Probability | What to do |
|---|---|---|
| 🟢 **LOW** | Below 20% | Continue normal operation. Keep monitoring. |
| 🟡 **MEDIUM** | 20%–50% | Schedule maintenance within the next shift. |
| 🟠 **HIGH** | 50%–80% | Prioritize inspection before the next production run. |
| 🔴 **CRITICAL** | Above 80% | Stop the machine immediately. Inspect now. |

These thresholds are set conservatively. If in doubt, escalate.

---

## 7. Submitting feedback (closing the loop)

The system gets smarter when you tell it whether its predictions were
right. After you've seen whether a machine actually failed (or not),
submit feedback:

1. Click **Predict** in the sidebar.
2. Switch to the **Feedback** tab.
3. Enter the **Prediction ID** from when you originally got the
   prediction.
4. Select what actually happened: **Failure occurred** or **No failure**.
5. Click **Submit**.

Your feedback is logged and the system's running accuracy gauge updates
immediately. Over time, this helps spot when the model needs retraining.

---

## 8. The Pipeline page — uploading new data

When you have a fresh batch of historical sensor data, you can upload
it to the system. This will trigger an automatic retraining run.

1. Click **Pipeline** in the sidebar.
2. Click **Browse files** and select your CSV (must match the standard
   schema described in §5).
3. Optionally type a reason for the retraining (e.g., "Q3 quarterly
   refresh").
4. Enter the **API key** your administrator provided.
5. Click **🚀 Upload & Store**.

You'll see a confirmation that the file was validated and stored.
Behind the scenes, this triggers the full training pipeline: data is
ingested, drift is checked, features are engineered, and a new model
is trained and registered. The whole process takes a few minutes.

If drift was detected in your uploaded data, you'll receive an email
notification once retraining is complete.

### Rolling back a model

If a newly trained model behaves badly, you can roll back to a
previous version from the same Pipeline page:

1. Scroll down to **Model Rollback**.
2. Choose a previous version from the dropdown.
3. Confirm with the API key.
4. Click **Roll back**.

The active model switches immediately. No restart required.

---

## 9. The Monitoring page — checking model health

Click **Monitoring** to see four things:

- **Model Quality** — current F1 score, accuracy, precision, recall,
  and a rolling accuracy gauge based on submitted feedback.
- **Drift Detection** — upload a CSV here to compare it against the
  data the model was trained on. The system tells you which features
  have shifted and by how much.
- **Retrain** — manual trigger to start a retraining run without
  uploading new data.
- **Alerts** — live status of the system's alert rules. If any are
  firing (drift detected, accuracy degraded, retraining storm), you'll
  see them here.

If you see drift flagged on more than 2–3 features, that's a strong
signal the model may be becoming stale. Consider uploading a fresh
dataset (see §8) or triggering a retraining run.

---

## 10. Failure types explained

The five failure types the system recognizes:

| Code | Name | What it means |
|---|---|---|
| TWF | Tool Wear Failure | The tool has been used beyond its safe wear limit |
| HDF | Heat Dissipation Failure | Temperature differential is too high for safe operation |
| PWF | Power Failure | Power output (torque × speed) is outside safe bounds |
| OSF | Overstrain Failure | Torque is excessive for the product quality grade |
| RNF | Random Failure | Stochastic, unpredictable failure (rare) |

The model predicts an overall **failure or no failure** outcome. To
identify the specific type, refer to the operating logs and known
patterns of the equipment.

---

## 11. Common problems and what to do

| What you see | What it means | What to do |
|---|---|---|
| "API Offline" on the dashboard | Backend service isn't responding | Contact your administrator |
| "Model not loaded" | The system can't find a trained model | Contact your administrator |
| "Drift detected" alert | The data you uploaded looks different from what the model was trained on | This is normal occasionally — review the drifted features under Monitoring |
| Upload rejected with "schema invalid" | Your CSV is missing required columns or has wrong column names | Check column names match exactly (see §5) |
| 401 error when uploading | Wrong or missing API key | Re-enter the key your administrator provided |
| Slow page loads | Server is busy with a training run | Wait a few minutes and retry |

---

## 12. Glossary

| Term | Plain-English meaning |
|---|---|
| **Drift** | When new data looks meaningfully different from training data |
| **F1 score** | A measure of model quality combining precision and recall (0–1, higher is better) |
| **Feedback** | You telling the system whether a prediction was correct |
| **Inference** | Asking the model to make a prediction |
| **Model version** | Each retrained model gets a new version number |
| **Pipeline** | The sequence of steps that prepare data and train a model |
| **Retrain** | Train a fresh model on more recent data |
| **Rollback** | Switch back to an older model version |

---

## 13. When to ask for help

Contact your administrator if any of these happen:

- The dashboard is unreachable or shows red status indicators for more
  than a few minutes.
- The "API Offline" message persists.
- A retraining run takes more than 30 minutes without finishing.
- You receive multiple alert emails in a short period.
- Predictions look obviously wrong (e.g., a clearly broken machine
  flagged as 0% risk, or a perfect machine flagged as critical).

For day-to-day usage questions, the User Manual page in the dashboard
itself is always the most up-to-date copy of this document.