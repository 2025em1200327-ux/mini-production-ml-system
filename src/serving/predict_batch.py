"""Batch scoring CLI: validate every row with the same schema as the API, score valid rows in one
vectorised call, write predictions keyed by customer id, and write rejected rows with the reason.

Run:  python -m src.serving.predict_batch --input data/new_customers.csv --output reports/scored.csv
Exit codes: 0 success, 1 input unreadable/missing columns, 2 success but some rows rejected.
"""
import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.config import FEATURES, ID_COL
from src.serving.model_service import ModelService
from src.serving.schemas import CustomerFeatures

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("predict_batch")


def validate_rows(df: pd.DataFrame):
    """Return (valid_frame, rejects_frame). Blank strings -> missing; extra columns (e.g. a label) ignored."""
    df = df.replace(r"^\s*$", np.nan, regex=True)
    good, bad = [], []
    for rec in df.to_dict(orient="records"):
        payload = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items() if k in FEATURES}
        payload["customer_id"] = None if rec.get(ID_COL) is None else str(rec.get(ID_COL))
        try:
            good.append(CustomerFeatures(**payload).model_dump())
        except ValidationError as e:
            reason = "; ".join(f"{'.'.join(map(str, x['loc'])) or 'row'}: {x['msg']}" for x in e.errors())
            bad.append({ID_COL: rec.get(ID_COL), "reason": reason})
    return pd.DataFrame(good), pd.DataFrame(bad, columns=[ID_COL, "reason"])


def score_file(input_path, output_path, rejects_path=None, service: ModelService | None = None) -> dict:
    service = service or ModelService()       # load time is reported by the service, not counted as throughput
    t0 = time.perf_counter()
    raw = pd.read_csv(input_path, dtype={ID_COL: str})
    missing = [c for c in FEATURES if c not in raw.columns]
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")
    valid, rejects = validate_rows(raw)
    t_val = time.perf_counter()
    out = pd.DataFrame(columns=[ID_COL, "churn_probability", "churn_flag", "decision", "model_version", "degraded"])
    if len(valid):
        probs, degraded, version = service.score(valid)
        flag = probs >= service.threshold
        out = pd.DataFrame({ID_COL: valid["customer_id"], "churn_probability": probs.round(6), "churn_flag": flag,
                            "decision": np.where(flag, "offer_retention", "no_action"),
                            "model_version": version, "degraded": degraded})
    out.to_csv(output_path, index=False)
    if rejects_path:
        rejects.to_csv(rejects_path, index=False)
    t_end = time.perf_counter()
    summary = {"rows_in": len(raw), "scored": len(out), "rejected": len(rejects),
               "seconds_total": round(t_end - t0, 3), "seconds_validation": round(t_val - t0, 3),
               "records_per_sec": round(len(raw) / (t_end - t0), 1), "model_version": service.model_version}
    log.info(json.dumps(summary))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score a CSV of customers for churn risk.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rejects", default=None, help="CSV for rows that failed validation (default: <output>.rejects.csv)")
    a = ap.parse_args(argv)
    try:
        s = score_file(a.input, a.output, a.rejects or str(a.output).replace(".csv", ".rejects.csv"))
    except (FileNotFoundError, ValueError) as e:
        log.error("batch failed: %s", e)
        return 1
    return 2 if s["rejected"] else 0


if __name__ == "__main__":
    sys.exit(main())
