"""Data- and prediction-drift checks against the training-time reference profile.

Run on any batch of production inputs (raw CSV or the API's PREDICTION_LOG JSONL):
    python -m src.monitoring --input data/sample_new_customers.csv --output reports/drift_latest.json
Exit code 2 when any check is at ALERT level, so a scheduler (cron, Airflow) can page or retrain.

Thresholds (industry convention for PSI, Siddiqi 2006; KS uses a p-value AND an effect size so that
large samples do not alert on trivial shifts):
    PSI < 0.10 stable | 0.10-0.25 WARN | >= 0.25 ALERT
    KS  ALERT if p < 0.01 and D >= 0.10
    Missing-rate change > 5 points: ALERT;  unseen categorical value: ALERT
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src import config as C
from src.data_pipeline import ChurnCleaner

EPS = 1e-4
PSI_WARN, PSI_ALERT, KS_P, KS_D, MISSING_DELTA = 0.10, 0.25, 0.01, 0.10, 0.05


def psi(ref: np.ndarray, cur: np.ndarray) -> float:
    ref, cur = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _bin_shares(values, edges):
    inner = np.asarray(edges)[1:-1]                    # open-ended outer bins catch out-of-range values
    idx = np.searchsorted(inner, values, side="right")
    return np.bincount(idx, minlength=len(inner) + 1) / max(len(values), 1)


def _level(p):
    return "ALERT" if p >= PSI_ALERT else ("WARN" if p >= PSI_WARN else "ok")


def drift_report(df: pd.DataFrame, reference: dict, model=None) -> dict:
    clean = ChurnCleaner().transform(df[C.FEATURES])
    rows = []
    for col, ref in reference["numeric"].items():
        cur = clean[col].dropna().to_numpy()
        ref_sample = np.asarray(ref["sample"])
        p = psi(_bin_shares(ref_sample, ref["edges"]), _bin_shares(cur, ref["edges"]))
        ks = ks_2samp(ref_sample, cur)
        miss_delta = float(clean[col].isna().mean() - ref["missing_rate"])
        status = "ALERT" if (p >= PSI_ALERT or (ks.pvalue < KS_P and ks.statistic >= KS_D)
                             or abs(miss_delta) > MISSING_DELTA) else _level(p)
        rows.append({"feature": col, "type": "numeric", "psi": round(p, 4), "ks_stat": round(float(ks.statistic), 4),
                     "ks_p": float(ks.pvalue), "missing_rate_change": round(miss_delta, 4),
                     "current_mean": round(float(cur.mean()), 3), "reference_mean": round(float(ref_sample.mean()), 3),
                     "status": status})
    for col, ref in reference["categorical"].items():
        cur = df[col].astype(str).value_counts(normalize=True).to_dict()
        cats = sorted(set(ref) | set(cur))
        p = psi(np.array([ref.get(c, 0.0) for c in cats]), np.array([cur.get(c, 0.0) for c in cats]))
        unseen = sorted(set(cur) - set(ref))
        rows.append({"feature": col, "type": "categorical", "psi": round(p, 4), "unseen_categories": unseen,
                     "status": "ALERT" if unseen else _level(p)})
    out = {"n_rows": len(df), "features": rows}
    if model is not None:                               # prediction drift: needs no labels
        scores = model.predict_proba(df[C.FEATURES])[:, 1]
        ref = reference["score"]
        p = psi(_bin_shares(np.asarray(ref["sample"]), ref["edges"]), _bin_shares(scores, ref["edges"]))
        out["prediction_drift"] = {"psi": round(p, 4), "mean_score": round(float(scores.mean()), 4),
                                   "reference_mean_score": round(float(np.mean(ref["sample"])), 4), "status": _level(p)}
    levels = [r["status"] for r in rows] + [out.get("prediction_drift", {}).get("status", "ok")]
    out["overall"] = "ALERT" if "ALERT" in levels else ("WARN" if "WARN" in levels else "ok")
    out["alerts"] = [r["feature"] for r in rows if r["status"] == "ALERT"]
    return out


def simulate_drift(df: pd.DataFrame, seed=C.SEED) -> pd.DataFrame:
    """A plausible market shock for the demo: a 20% price rise, a promotion that moves 30% of
    contract customers onto month-to-month plans, and a burst of new sign-ups (40% of rows get tenure 0-6)."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["MonthlyCharges"] = d["MonthlyCharges"] * 1.20
    tc = pd.to_numeric(d["TotalCharges"], errors="coerce")
    d["TotalCharges"] = tc * 1.20
    move = (d["Contract"] != "Month-to-month") & (rng.random(len(d)) < 0.30)
    d.loc[move, "Contract"] = "Month-to-month"
    new = rng.random(len(d)) < 0.40
    d.loc[new, "tenure"] = rng.integers(0, 7, new.sum())
    d.loc[new, "TotalCharges"] = d.loc[new, "tenure"] * d.loc[new, "MonthlyCharges"]
    return d


def load_input(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True) if path.endswith(".jsonl") else pd.read_csv(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Drift check against the training reference profile.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=str(C.REPORT_DIR / "drift_latest.json"))
    a = ap.parse_args(argv)
    reference = json.loads(C.REFERENCE_FILE.read_text())
    report = drift_report(load_input(a.input), reference, joblib.load(C.MODEL_FILE))
    Path(a.output).write_text(json.dumps(report, indent=2))
    print(f"overall: {report['overall']}  alerts: {report['alerts']}")
    return 2 if report["overall"] == "ALERT" else 0


if __name__ == "__main__":
    sys.exit(main())
