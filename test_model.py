import pytest

from anomaly_model import (
    DEMO_SENSORS,
    MODELS,
    AnomalyDetector,
    add_alerts,
    evaluate,
    generate_sensor_data,
)


@pytest.mark.parametrize("model", MODELS)
def test_detects_injected_failures(model):
    df = generate_sensor_data(n_points=3000, seed=1)
    n_train = int(len(df) * 0.25)
    det = AnomalyDetector(model, contamination=0.02, window=6).fit(df.iloc[:n_train], DEMO_SENSORS)
    pred = det.predict(df)
    alerts = add_alerts(pred["anomaly"], k=3, m=6)

    test = df.iloc[n_train:]
    scores = evaluate(test["is_anomaly"], alerts.iloc[n_train:])
    assert scores["recall"] > 0.5
    assert scores["precision"] > 0.5
    assert pred["risk"].between(0, 1).all()


def test_top_driver_is_a_sensor():
    df = generate_sensor_data(n_points=1500, seed=3)
    det = AnomalyDetector().fit(df.iloc[:400], DEMO_SENSORS)
    assert set(det.predict(df)["top_driver"]).issubset(set(DEMO_SENSORS))


def test_requires_sensors():
    df = generate_sensor_data(n_points=500)
    with pytest.raises(ValueError):
        AnomalyDetector().fit(df, [])
