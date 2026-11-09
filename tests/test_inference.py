"""Inference integrity: the served artifact IS the evaluated model, and serving == offline scoring."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src import config as C
from src.data_pipeline import ChurnCleaner, load_raw, split
from src.serving.predict_batch import score_file
from tests.conftest import VALID


def _frame(p):
    return pd.DataFrame([{k: p[k] for k in C.FEATURES}])


def test_artifact_reproduces_reported_test_metric():
    model = joblib.load(C.MODEL_FILE)
    _, _, test = split(load_raw())
    reported = json.loads((C.REPORT_DIR / "test_metrics.json").read_text())
    pr = average_precision_score(test[C.TARGET_COL], model.predict_proba(test[C.FEATURES])[:, 1])
    assert abs(pr - reported["test"][reported["chosen"]]["pr_auc"]) < 1e-9


def test_api_equals_offline_scoring(client):
    offline = joblib.load(C.MODEL_FILE).predict_proba(_frame(VALID))[0, 1]
    online = client.post("/predict", json=VALID).json()["churn_probability"]
    assert abs(online - offline) < 1e-6          # no training/serving skew


def test_deterministic(client):
    a = client.post("/predict", json=VALID).json()["churn_probability"]
    b = client.post("/predict", json=VALID).json()["churn_probability"]
    assert a == b


def test_extreme_but_valid_values_score_finite(client):
    r = client.post("/predict", json={**VALID, "tenure": 120, "MonthlyCharges": 1000, "TotalCharges": 200000})
    assert r.status_code == 200 and np.isfinite(r.json()["churn_probability"])


def test_pipeline_tolerates_unseen_category_if_schema_bypassed():
    """Defence in depth: even if a value slipped past the schema, the encoder ignores it instead of crashing."""
    model = joblib.load(C.MODEL_FILE)
    p = model.predict_proba(_frame({**VALID, "PaymentMethod": "Crypto"}))[0, 1]
    assert 0 <= p <= 1


def test_new_customer_blank_total_charges_becomes_zero():
    df = _frame({**VALID, "tenure": 0, "TotalCharges": None})
    assert ChurnCleaner().transform(df)["TotalCharges"].iloc[0] == 0.0


def test_batch_cli_rejects_and_keeps_ids(tmp_path):
    rows = pd.DataFrame([{**{k: VALID[k] for k in C.FEATURES}, C.ID_COL: "A"},
                         {**{k: VALID[k] for k in C.FEATURES}, C.ID_COL: "B", "Contract": "Three year"},
                         {**{k: VALID[k] for k in C.FEATURES}, C.ID_COL: "C", "TotalCharges": " "}])
    src = tmp_path / "in.csv"; rows.to_csv(src, index=False)
    s = score_file(src, tmp_path / "out.csv", tmp_path / "rej.csv")
    out, rej = pd.read_csv(tmp_path / "out.csv"), pd.read_csv(tmp_path / "rej.csv")
    assert s["scored"] == 2 and s["rejected"] == 1
    assert list(out[C.ID_COL]) == ["A", "C"] and rej[C.ID_COL].tolist() == ["B"]
