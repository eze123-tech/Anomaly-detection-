"""Streamlit dashboard for equipment anomaly detection / predictive maintenance."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from anomaly_model import (
    DEMO_SENSORS,
    MODELS,
    AnomalyDetector,
    add_alerts,
    evaluate,
    generate_sensor_data,
    risk_status,
)

st.set_page_config(page_title="Equipment Anomaly Detection", page_icon="🛠️", layout="wide")
st.title("🛠️ Equipment Anomaly Detection")
st.caption(
    "Unsupervised machine learning that learns what healthy equipment looks like "
    "and flags abnormal sensor behaviour before it becomes a breakdown."
)


@st.cache_data
def load_demo(n_points: int, seed: int) -> pd.DataFrame:
    return generate_sensor_data(n_points=n_points, seed=seed)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1. Data")
    source = st.radio("Source", ["Demo data", "Upload CSV"])
    label_col = None
    time_col = None

    if source == "Demo data":
        n_points = st.slider("Number of readings", 1000, 8000, 3000, step=500)
        seed = st.number_input("Random seed", value=42, step=1)
        df = load_demo(n_points, int(seed))
        time_col, label_col = "timestamp", "is_anomaly"
        sensor_options = DEMO_SENSORS
        default_sensors = DEMO_SENSORS
    else:
        up = st.file_uploader("CSV with sensor readings", type="csv")
        if up is None:
            st.info("Upload a CSV, or switch to Demo data.")
            st.stop()
        df = pd.read_csv(up)
        none = "(none)"
        tc = st.selectbox("Timestamp column (optional)", [none] + list(df.columns))
        lc = st.selectbox("Known-failure label column (optional, 0/1)", [none] + list(df.columns))
        time_col = None if tc == none else tc
        label_col = None if lc == none else lc
        numeric = [c for c in df.select_dtypes("number").columns if c != label_col]
        sensor_options = numeric
        default_sensors = numeric[:8]

    sensors = st.multiselect("Sensor columns", sensor_options, default=default_sensors)

    st.header("2. Model")
    model_name = st.selectbox("Algorithm", MODELS)
    train_frac = st.slider("Healthy training share of data", 0.10, 0.70, 0.25, 0.05,
                           help="The first part of the data, assumed to be normal operation.")
    contamination = st.slider("Expected anomaly rate", 0.005, 0.10, 0.02, 0.005,
                              help="Sets the sensitivity: higher flags more readings.")
    window = st.slider("Rolling window (readings)", 1, 24, 6)

    st.header("3. Alerting")
    m = st.slider("Look at last M readings", 2, 20, 6)
    k = st.slider("Alert if K of them are anomalous", 1, m, min(3, m))

if time_col:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)
    x_axis = df[time_col]
else:
    x_axis = pd.Series(range(len(df)), name="reading")

if not sensors:
    st.warning("Select at least one sensor column in the sidebar.")
    st.stop()

n_train = int(len(df) * train_frac)
if n_train < 50:
    st.warning("Not enough data to train: use at least 50 healthy readings.")
    st.stop()

# -------------------------------------------------------------------- model
with st.spinner("Training model..."):
    detector = AnomalyDetector(model_name, contamination, window).fit(df.iloc[:n_train], sensors)
    pred = detector.predict(df)
pred["alert"] = add_alerts(pred["anomaly"], k, m)
res = pd.concat([df, pred], axis=1)

# ------------------------------------------------------------------ summary
last = pred.iloc[-1]
status = risk_status(float(last["risk"]), bool(last["alert"]))
icon = {"Healthy": "🟢", "Warning": "🟠", "Critical": "🔴"}[status]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current status", f"{icon} {status}")
c2.metric("Latest risk score", f"{last['risk']:.0%}")
c3.metric("Anomalous readings", int(pred["anomaly"].sum()))
c4.metric("Alert readings", int(pred["alert"].sum()))
if status != "Healthy":
    st.error(f"Largest deviation right now: **{last['top_driver']}**. Inspect before it fails.") \
        if status == "Critical" else st.warning(f"Largest deviation right now: **{last['top_driver']}**.")

if label_col:
    test = res.iloc[n_train:]
    point = evaluate(test[label_col], test["anomaly"])
    alert = evaluate(test[label_col], test["alert"])
    st.subheader("Accuracy against known failures (test period only)")
    a, b = st.columns(2)
    a.write("**Per-reading flags**")
    a.write(f"Precision {point['precision']:.2f} · Recall {point['recall']:.2f} · F1 {point['f1']:.2f}")
    b.write("**After alert rule**")
    b.write(f"Precision {alert['precision']:.2f} · Recall {alert['recall']:.2f} · F1 {alert['f1']:.2f}")

# ------------------------------------------------------------------- charts
st.subheader("Sensor readings and detected anomalies")
rows = len(sensors) + 1
fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                    subplot_titles=[*sensors, "Risk score"])
flagged = res["anomaly"] == 1
for i, s in enumerate(sensors, start=1):
    fig.add_trace(go.Scatter(x=x_axis, y=res[s], mode="lines", name=s,
                             line=dict(width=1)), row=i, col=1)
    fig.add_trace(go.Scatter(x=x_axis[flagged], y=res.loc[flagged, s], mode="markers",
                             marker=dict(color="red", size=5), name="anomaly",
                             showlegend=(i == 1)), row=i, col=1)
fig.add_trace(go.Scatter(x=x_axis, y=res["risk"], mode="lines", name="risk",
                         line=dict(color="darkorange")), row=rows, col=1)
fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=rows, col=1)
fig.add_vline(x=x_axis.iloc[n_train], line_dash="dot", line_color="gray")
fig.update_layout(height=220 * rows, margin=dict(t=40, b=10), legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)
st.caption("Dotted vertical line: end of the healthy training period.")

# ------------------------------------------------------------------- tables
st.subheader("Highest-risk readings")
show = [c for c in [time_col, "risk", "top_driver", "alert", *sensors] if c]
st.dataframe(res.sort_values("risk", ascending=False).head(25)[show], use_container_width=True)

st.download_button("Download full results (CSV)", res.to_csv(index=False).encode(),
                   "anomaly_results.csv", "text/csv")

with st.expander("How it works"):
    st.markdown(
        """
        1. **Features**: each sensor's value, rolling mean, rolling spread and rate of change.
        2. **Training**: the model learns normal behaviour from the healthy period (no failure labels needed).
        3. **Scoring**: every reading gets a risk score; 50% is the decision threshold.
        4. **Alert rule**: an alert fires only if K of the last M readings are anomalous,
           which filters one-off spikes and catches sustained degradation.
        5. **Root cause hint**: the sensor furthest from its normal range is reported as the top driver.
        """
    )
