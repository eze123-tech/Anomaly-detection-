# Equipment Anomaly Detection

Predictive-maintenance dashboard built with Python, scikit-learn and Streamlit.
It learns what *healthy* equipment looks like from sensor data and flags abnormal
behaviour early, so maintenance can happen before a breakdown.

## Features
- Three unsupervised algorithms: Isolation Forest, Local Outlier Factor, One-Class SVM
- Feature engineering: rolling mean, rolling std and rate of change per sensor
- 0-100% risk score and Healthy / Warning / Critical status
- Alert rule ("K of the last M readings are anomalous") to reduce false alarms
- Root-cause hint: which sensor deviates most
- Demo data with simulated degradation, or upload your own CSV
- Optional accuracy report if your CSV has a 0/1 failure label column
- Download results as CSV

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Run the tests:
```bash
pip install pytest
pytest
```

## Your own data
Upload a CSV with one row per reading and numeric sensor columns, e.g.:

| timestamp | temperature | vibration | pressure |
|-----------|-------------|-----------|----------|
| 2026-01-01 00:00 | 70.2 | 0.51 | 101.3 |

Important: the first part of the file (default 25%) should be normal operation.
The model treats it as the definition of "healthy".

## Deploy with GitHub + Streamlit Community Cloud
1. Create a new repository on GitHub (e.g. `equipment-anomaly-detection`).
2. Push this project:
   ```bash
   git init
   git add .
   git commit -m "Equipment anomaly detection app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/equipment-anomaly-detection.git
   git push -u origin main
   ```
3. Go to https://share.streamlit.io and sign in with GitHub.
4. Click **Create app**, pick the repository and branch `main`, set the main file to `app.py`, then **Deploy**.
5. Every `git push` to `main` redeploys the app automatically.

## Project layout
```
app.py              Streamlit dashboard
anomaly_model.py    Data simulation, features, detector, alerts, metrics
test_model.py       Tests
requirements.txt    Dependencies
```
