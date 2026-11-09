"""Score every candidate ONCE on the untouched test partition and produce the report's tables/figures.

Run:  python -m src.evaluate      (after python -m src.train)
"""
import json
import logging

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score)

from src import config as C
from src.data_pipeline import load_raw, save_json, split
from src.train import best_threshold, net_benefit, rule_flag

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evaluate")
NAMES = {"dummy_prior": "Dummy (prior)", "logistic_weighted_calibrated": "Logistic (weighted, calibrated)",
         "lightgbm_unweighted": "LightGBM (unweighted)", "lightgbm_weighted": "LightGBM (weighted)",
         "lightgbm_weighted_calibrated": "LightGBM (weighted, calibrated)"}


def recall_at_precision(y, p, min_precision=0.60):
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= min_precision
    return float(rec[ok].max()) if ok.any() else 0.0


def bootstrap_ci(fn, y, p, n=2000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    vals = [fn(y[i], p[i]) for i in idx]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test = split(load_raw())
    val_res = json.loads((C.REPORT_DIR / "validation_results.json").read_text())
    meta = json.loads(C.METADATA_FILE.read_text())
    chosen, thr = meta["model_name"], meta["threshold"]
    Xte, yte, mc = test[C.FEATURES], test[C.TARGET_COL].to_numpy(), test["MonthlyCharges"].to_numpy()

    probs, table = {}, {}
    for name in NAMES:
        m = joblib.load(C.ARTIFACT_DIR / "candidates" / f"{name}.joblib")
        p = m.predict_proba(Xte)[:, 1]
        probs[name] = p
        t = val_res["results"][name]["threshold"]          # threshold fixed on VALIDATION
        flag = p >= t
        table[name] = {"pr_auc": average_precision_score(yte, p), "roc_auc": roc_auc_score(yte, p),
                       "brier": brier_score_loss(yte, p), "recall_at_precision_0.60": recall_at_precision(yte, p),
                       "threshold_from_val": t, "f1_macro": f1_score(yte, flag, average="macro"),
                       "precision": float(yte[flag].mean()) if flag.any() else 0.0,
                       "recall": float(flag[yte == 1].mean()), "flag_rate": float(flag.mean()),
                       "net_benefit_usd": net_benefit(yte, flag, mc)}
    rf = rule_flag(test)
    table["heuristic_rule"] = {"pr_auc": None, "roc_auc": None, "brier": None, "recall_at_precision_0.60": None,
                               "threshold_from_val": None, "f1_macro": f1_score(yte, rf, average="macro"),
                               "precision": float(yte[rf].mean()), "recall": float(rf[yte == 1].mean()),
                               "flag_rate": float(rf.mean()), "net_benefit_usd": net_benefit(yte, rf, mc)}
    table["contact_everyone"] = {"net_benefit_usd": net_benefit(yte, np.ones_like(yte, bool), mc),
                                 "flag_rate": 1.0, "recall": 1.0, "precision": float(yte.mean())}

    # Uncertainty: 95% bootstrap intervals and a paired comparison of the two real contenders
    pf = probs[chosen]
    ci = {"pr_auc": bootstrap_ci(average_precision_score, yte, pf)}
    rng = np.random.default_rng(C.SEED)
    idx = rng.integers(0, len(yte), size=(2000, len(yte)))
    nb = [net_benefit(yte[i], pf[i] >= thr, mc[i]) for i in idx]
    ci["net_benefit_usd"] = (float(np.percentile(nb, 2.5)), float(np.percentile(nb, 97.5)))
    other = "logistic_weighted_calibrated" if chosen != "logistic_weighted_calibrated" else "lightgbm_weighted_calibrated"
    diff = [average_precision_score(yte[i], pf[i]) - average_precision_score(yte[i], probs[other][i]) for i in idx]
    paired = {"vs": other, "mean_pr_auc_diff": float(np.mean(diff)),
              "ci95": (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))),
              "share_of_resamples_chosen_better": float(np.mean(np.array(diff) > 0))}

    # Confusion matrix of the deployed model at the deployed threshold
    tn, fp, fn, tp = confusion_matrix(yte, pf >= thr).ravel()

    # Sensitivity of the cost-optimal threshold to the business assumptions (chosen on VALIDATION)
    pv = joblib.load(C.MODEL_FILE).predict_proba(val[C.FEATURES])[:, 1]
    sens = []
    for cost in (25, 50, 100):
        for succ in (0.2, 0.3, 0.5):
            t, v, _, _ = best_threshold(val[C.TARGET_COL].to_numpy(), pv, val["MonthlyCharges"].to_numpy(),
                                        offer_cost=cost, success=succ)
            sens.append({"offer_cost": cost, "success_rate": succ, "best_threshold": t, "val_net_benefit_usd": v})

    # Permutation importance on VALIDATION (whole pipeline, raw input columns, PR-AUC drop)
    final = joblib.load(C.MODEL_FILE)
    pi = permutation_importance(final, val[C.FEATURES], val[C.TARGET_COL], scoring="average_precision",
                                n_repeats=10, random_state=C.SEED, n_jobs=1)
    imp = sorted(zip(C.FEATURES, pi.importances_mean, pi.importances_std), key=lambda r: -r[1])

    out = {"chosen": chosen, "threshold": thr, "test": table, "bootstrap_ci95": ci, "paired_bootstrap": paired,
           "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
           "threshold_sensitivity": sens,
           "permutation_importance_val": [{"feature": f, "pr_auc_drop": float(m), "std": float(s)} for f, m, s in imp]}
    save_json(out, C.REPORT_DIR / "test_metrics.json")

    # ---------------- Figures ----------------
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for name, style in [("lightgbm_weighted_calibrated", "-"), ("logistic_weighted_calibrated", "--")]:
        prec, rec, _ = precision_recall_curve(yte, probs[name])
        ax.plot(rec, prec, style, label=f"{NAMES[name]}  (PR-AUC {table[name]['pr_auc']:.3f})")
    # a constant score has no ranking: its PR "curve" is the flat line at the churn rate
    ax.axhline(yte.mean(), ls=":", c="grey", label=f"{NAMES['dummy_prior']}  (PR-AUC {table['dummy_prior']['pr_auc']:.3f})")
    ax.scatter([table["heuristic_rule"]["recall"]], [table["heuristic_rule"]["precision"]], marker="s", c="k",
               label="Heuristic rule (one operating point)", zorder=5)
    ax.scatter([table[chosen]["recall"]], [table[chosen]["precision"]], marker="o", s=70, facecolors="none",
               edgecolors="C3", linewidths=2, label=f"Deployed threshold {thr:.2f}", zorder=6)
    ax.set_xlabel("Recall (share of churners caught)"); ax.set_ylabel("Precision (share of flags that churn)")
    ax.set_title("Precision-recall on the test set"); ax.legend(fontsize=7.5, loc="upper right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "pr_curves.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    grid = val_res["threshold_grid"]
    ax.plot(grid, np.array(val_res["net_benefit_curves"][chosen]) / 1000, label=NAMES[chosen])
    ax.plot(grid, np.array(val_res["net_benefit_curves"]["logistic_weighted_calibrated"]) / 1000, "--",
            label=NAMES["logistic_weighted_calibrated"])
    ax.axhline(val_res["results"]["heuristic_rule"]["net_benefit_usd"] / 1000, c="k", ls=":", label="Heuristic rule")
    ax.axhline(val_res["results"]["contact_everyone"]["net_benefit_usd"] / 1000, c="grey", ls="-.", label="Contact everyone")
    ax.axvline(thr, c="C3", lw=1); ax.text(thr + 0.01, ax.get_ylim()[0] + 2, f"chosen {thr:.2f}", color="C3", fontsize=8)
    ax.set_xlabel("Decision threshold on churn probability"); ax.set_ylabel("Net benefit on validation (USD thousands)")
    ax.set_title("Choosing the threshold by business value"); ax.legend(fontsize=7.5); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "net_benefit.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for name, lab in [("lightgbm_weighted", "Weighted, uncalibrated"), ("lightgbm_weighted_calibrated", "Weighted + Platt calibration")]:
        fy, mp = calibration_curve(yte, probs[name], n_bins=10, strategy="quantile")
        ax.plot(mp, fy, "o-", label=f"{lab} (Brier {table[name]['brier']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_xlabel("Predicted churn probability"); ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration on the test set"); ax.legend(fontsize=7.5); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "calibration.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    top = imp[:10][::-1]
    ax.barh([f for f, _, _ in top], [m for _, m, _ in top], xerr=[s for _, _, s in top], color="#2F6DB5")
    ax.set_xlabel("Drop in validation PR-AUC when the column is shuffled"); ax.set_title("Permutation importance (top 10)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "importance.png", dpi=180); plt.close(fig)

    log.info("test PR-AUC %.4f (95%% CI %.3f-%.3f) | net benefit $%.0f", table[chosen]["pr_auc"], *ci["pr_auc"],
             table[chosen]["net_benefit_usd"])
    print(json.dumps(out, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
