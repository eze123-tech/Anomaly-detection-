"""Core logic for equipment anomaly detection.

Pipeline:
    1. Sensor readings (temperature, vibration, ...) -> engineered features
       (raw value, rolling mean, rolling std, first difference).
    2. An unsupervised model is trained on a period assumed to be healthy.
    3. Every new reading gets a risk score (0-1), an anomaly flag, and the
       sensor that deviates most from normal behaviour.
    4. An alert is raised only when several recent readings are anomalous,
       which cuts false alarms from one-off noise spikes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

MODELS = ["Isolation Forest", "Local Outlier Factor", "One-Class SVM"]
DEMO_SENSORS = ["temperature", "vibration", "pressure", "rpm", "current"]


# --------------------------------------------------------------------------
# Demo data
# --------------------------------------------------------------------------
def generate_sensor_data(
    n_points: int = 3000,
    n_failures: int = 4,
    seed: int = 42,
    freq: str = "10min",
) -> pd.DataFrame:
    """Simulate a machine with daily cycles and gradual pre-failure degradation.

    Returns a DataFrame with a timestamp, five sensors and an ``is_anomaly``
    ground-truth label (1 while the machine is degrading).
    """
    rng = np.random.default_rng(seed)
    x = np.arange(n_points)
    daily = np.sin(2 * np.pi * x / 144)  # 144 x 10min = 1 day

    temperature = 70 + 3 * daily + rng.normal(0, 1.0, n_points)
    vibration = 0.5 + 0.05 * daily + rng.normal(0, 0.05, n_points)
    pressure = 100 + 2 * daily + rng.normal(0, 1.0, n_points)
    rpm = 1500 + 20 * daily + rng.normal(0, 10, n_points)
    current = 12 + 0.01 * (rpm - 1500) + rng.normal(0, 0.3, n_points)
    label = np.zeros(n_points, dtype=int)

    # Failure episodes occur only in the later part of the series so the early
    # part can be used as clean training data.
    start_min = int(n_points * 0.35)
    slots = np.linspace(start_min, n_points - 60, n_failures).astype(int)
    for s in slots:
        length = int(rng.integers(25, 45))
        e = min(s + length, n_points)
        ramp = np.linspace(0.2, 1.0, e - s)
        temperature[s:e] += 14 * ramp
        vibration[s:e] += 0.6 * ramp + rng.normal(0, 0.08, e - s)
        current[s:e] += 3.0 * ramp
        pressure[s:e] -= 6 * ramp
        label[s:e] = 1

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n_points, freq=freq),
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "rpm": rpm,
            "current": current,
            "is_anomaly": label,
        }
    )


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------
def _make_model(name: str, contamination: float, random_state: int, n_features: int = 20):
    if name == "Isolation Forest":
        return IsolationForest(
            n_estimators=200, contamination="auto", random_state=random_state, n_jobs=-1
        )
    if name == "Local Outlier Factor":
        return LocalOutlierFactor(n_neighbors=35, novelty=True)
    if name == "One-Class SVM":
        nu = float(np.clip(contamination, 0.001, 0.5))
        # A smooth kernel (gamma well below sklearn's "scale" default) avoids
        # flagging normal day/night variation as anomalous.
        return OneClassSVM(kernel="rbf", gamma=0.1 / max(1, n_features), nu=nu)
    raise ValueError(f"Unknown model: {name}")


class AnomalyDetector:
    """Unsupervised detector trained on (assumed) healthy equipment data."""

    def __init__(
        self,
        model: str = "Isolation Forest",
        contamination: float = 0.02,
        window: int = 6,
        random_state: int = 42,
    ):
        self.model_name = model
        self.contamination = contamination
        self.window = max(1, int(window))
        self.random_state = random_state

    # -- features ----------------------------------------------------------
    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        base = df[self.sensors].astype(float).ffill().bfill()
        roll = base.rolling(self.window, min_periods=1)
        parts = [
            base,
            roll.mean().add_suffix("_mean"),
            roll.std().fillna(0).add_suffix("_std"),
            base.diff().fillna(0).add_suffix("_diff"),
        ]
        return pd.concat(parts, axis=1)

    def _raw_score(self, X_scaled: np.ndarray) -> np.ndarray:
        # sklearn: higher decision_function = more normal. Flip so higher = worse.
        return -self.model_.decision_function(X_scaled)

    # -- API ---------------------------------------------------------------
    def fit(self, df: pd.DataFrame, sensors: list[str]) -> "AnomalyDetector":
        if not sensors:
            raise ValueError("Select at least one sensor column.")
        self.sensors = list(sensors)
        clean = df[self.sensors].astype(float).ffill().bfill()
        self.mean_ = clean.mean()
        self.std_ = clean.std().replace(0, 1.0).fillna(1.0)

        X = self._features(df)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.model_ = _make_model(
            self.model_name, self.contamination, self.random_state, Xs.shape[1]
        )
        self.model_.fit(Xs)

        raw = self._raw_score(Xs)
        self.threshold_ = float(np.quantile(raw, 1 - self.contamination))
        self.scale_ = float(raw.std()) or 1.0
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return risk (0-1), anomaly flag and the most deviating sensor."""
        Xs = self.scaler_.transform(self._features(df))
        raw = self._raw_score(Xs)
        z = (raw - self.threshold_) / self.scale_
        risk = 1.0 / (1.0 + np.exp(-2.0 * z))  # 0.5 exactly at the threshold

        base = df[self.sensors].astype(float).ffill().bfill()
        deviation = ((base - self.mean_) / self.std_).abs()
        out = pd.DataFrame(index=df.index)
        out["risk"] = risk
        out["anomaly"] = (raw > self.threshold_).astype(int)
        out["top_driver"] = deviation.idxmax(axis=1)
        return out


# --------------------------------------------------------------------------
# Alerts and evaluation
# --------------------------------------------------------------------------
def add_alerts(flags: pd.Series, k: int = 3, m: int = 6) -> pd.Series:
    """Alert when at least ``k`` of the last ``m`` readings are anomalous."""
    return (flags.rolling(m, min_periods=1).sum() >= k).astype(int)


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def risk_status(risk: float, alert: bool) -> str:
    if alert:
        return "Critical"
    if risk >= 0.5:
        return "Warning"
    return "Healthy"
