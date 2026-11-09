import numpy as np


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["model_loaded"]


def test_predict_contract(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_flag"] == (body["churn_probability"] >= body["threshold"])
    assert body["decision"] == ("offer_retention" if body["churn_flag"] else "no_action")
    assert body["customer_id"] == "TEST-0001" and body["degraded"] is False
    assert "X-Request-ID" in r.headers and "X-Process-Time-ms" in r.headers


def test_validation_error_shape(client, payload):
    r = client.post("/predict", json={**payload, "Contract": "Three year"})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_error" and body["detail"][0]["field"] == "Contract"


def test_malformed_json(client):
    r = client.post("/predict", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_batch_ok_and_too_large(client, payload):
    r = client.post("/predict/batch", json={"records": [payload] * 3})
    assert r.status_code == 200 and r.json()["n"] == 3
    r = client.post("/predict/batch", json={"records": [payload] * 1001})
    assert r.status_code == 413 and r.json()["error"] == "batch_too_large"


def test_fallback_when_model_missing(client, payload):
    svc = client.app.state.service
    saved, svc.model = svc.model, None
    try:
        assert client.get("/health").json()["status"] == "degraded"
        body = client.post("/predict", json=payload).json()
        assert body["degraded"] is True and body["model_version"] == "fallback-rule-v1"
        assert body["churn_probability"] == svc.fallback["p_if_rule"]     # month-to-month, tenure 5
    finally:
        svc.model = saved


def test_fallback_when_scoring_raises(client, payload):
    svc = client.app.state.service

    class Broken:
        def predict_proba(self, X):
            raise RuntimeError("simulated scoring failure")

    saved, svc.model = svc.model, Broken()
    try:
        r = client.post("/predict", json=payload)
        assert r.status_code == 200 and r.json()["degraded"] is True
    finally:
        svc.model = saved


def test_fallback_on_non_finite_scores(client, payload):
    svc = client.app.state.service

    class NaNModel:
        def predict_proba(self, X):
            return np.full((len(X), 2), np.nan)

    saved, svc.model = svc.model, NaNModel()
    try:
        assert client.post("/predict", json=payload).json()["degraded"] is True
    finally:
        svc.model = saved
