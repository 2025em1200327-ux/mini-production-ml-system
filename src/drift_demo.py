"""Demonstrates the monitor on three batches: validation and test (should be stable) and a simulated
market shock (should alert). Writes reports/drift_demo.json and reports/figures/drift_psi.png."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as C
from src.data_pipeline import load_raw, save_json, split
from src.monitoring import PSI_ALERT, PSI_WARN, drift_report, simulate_drift


def main():
    _, val, test = split(load_raw())
    ref = json.loads(C.REFERENCE_FILE.read_text())
    model = joblib.load(C.MODEL_FILE)
    scen = {"validation (no change)": val, "test (no change)": test, "simulated market shock": simulate_drift(test)}
    out = {k: drift_report(d, ref, model) for k, d in scen.items()}
    save_json(out, C.REPORT_DIR / "drift_demo.json")
    feats = [r["feature"] for r in out["test (no change)"]["features"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(feats) + 1)
    for i, (k, c) in enumerate([("test (no change)", "#2F6DB5"), ("simulated market shock", "#C2571A")]):
        vals = [r["psi"] for r in out[k]["features"]] + [out[k]["prediction_drift"]["psi"]]
        ax.bar(x + (i - 0.5) * 0.4, vals, 0.4, label=k, color=c)
    ax.axhline(PSI_WARN, ls=":", c="grey"); ax.axhline(PSI_ALERT, ls="--", c="k")
    ax.text(len(x) - 0.5, PSI_ALERT + 0.01, "alert 0.25", ha="right", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(feats + ["MODEL SCORE"], rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel("Population Stability Index"); ax.set_title("Drift monitor: stable batch vs simulated shock")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(C.FIG_DIR / "drift_psi.png", dpi=180)
    for k, r in out.items():
        print(f"{k:26s} overall={r['overall']:5s} alerts={r['alerts']} score_psi={r['prediction_drift']['psi']}")


if __name__ == "__main__":
    main()
