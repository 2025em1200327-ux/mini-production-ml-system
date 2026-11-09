"""Train baselines and candidate models on TRAIN, select on VALIDATION, export artifacts.

Run:  python -m src.train
Outputs: artifacts/churn_model.joblib, artifacts/model_metadata.json, artifacts/reference_profile.json,
         artifacts/candidates/*.joblib (for the comparison in evaluate.py), reports/validation_results.json
"""
import json
import logging
import platform
import time
from datetime import datetime, timezone

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src import config as C
from src.data_pipeline import ChurnCleaner, leakage_audit, load_raw, make_pipeline, save_json, split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")
CV = StratifiedKFold(5, shuffle=True, random_state=C.SEED)


# ----------------------------------------------------------------------------------------------
# Business value of a decision (used for threshold selection and reported in evaluate.py)
# ----------------------------------------------------------------------------------------------
def net_benefit(y, flag, monthly_charges, offer_cost=C.OFFER_COST, success=C.OFFER_SUCCESS_RATE,
                horizon=C.VALUE_HORIZON_MONTHS) -> float:
    """USD gained by contacting the flagged customers, relative to contacting nobody.
    Each contact costs `offer_cost`; a contacted true churner stays with probability `success`,
    saving `horizon` months of that customer's own MonthlyCharges."""
    flag, y = np.asarray(flag, bool), np.asarray(y, bool)
    saved = success * horizon * np.asarray(monthly_charges)[flag & y].sum()
    return float(saved - offer_cost * flag.sum())


def best_threshold(y, p, monthly_charges, **kw):
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    values = [net_benefit(y, p >= t, monthly_charges, **kw) for t in grid]
    i = int(np.argmax(values))
    return float(grid[i]), float(values[i]), grid.tolist(), values


# ----------------------------------------------------------------------------------------------
# Heuristic baseline - also the serving fallback
# ----------------------------------------------------------------------------------------------
def rule_flag(df: pd.DataFrame) -> np.ndarray:
    """Retention-team rule of thumb: month-to-month contract AND under 12 months of tenure."""
    return ((df["Contract"] == "Month-to-month") & (df["tenure"] < 12)).to_numpy()


def fallback_table(train: pd.DataFrame) -> dict:
    """Churn rate inside / outside the rule on TRAIN - the fallback's probability for each group."""
    f = rule_flag(train)
    return {"rule": "Contract == 'Month-to-month' and tenure < 12",
            "p_if_rule": round(float(train.loc[f, C.TARGET_COL].mean()), 4),
            "p_otherwise": round(float(train.loc[~f, C.TARGET_COL].mean()), 4)}


def metrics(y, p, thr):
    flag = p >= thr
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "f1_macro": f1_score(y, flag, average="macro"),
            "precision": float((y[flag] == 1).mean()) if flag.any() else 0.0,
            "recall": float(flag[y == 1].mean()), "flag_rate": float(flag.mean())}


def reference_profile(train, model) -> dict:
    """Training-time distribution snapshot used by monitoring.py for drift checks (PSI / KS)."""
    clean = ChurnCleaner().transform(train[C.FEATURES])
    prof = {"numeric": {}, "categorical": {}, "n_rows": len(train)}
    rng = np.random.default_rng(C.SEED)
    for col in C.NUMERIC_FEATURES:
        v = clean[col].dropna().to_numpy()
        edges = np.unique(np.quantile(v, np.linspace(0, 1, 11)))
        prof["numeric"][col] = {"edges": edges.tolist(),
                                "sample": rng.choice(v, size=min(2000, len(v)), replace=False).round(4).tolist(),
                                "missing_rate": float(clean[col].isna().mean())}
    for col in C.CATEGORICAL_FEATURES + C.BINARY_NUMERIC:
        prof["categorical"][col] = train[col].astype(str).value_counts(normalize=True).round(6).to_dict()
    scores = model.predict_proba(train[C.FEATURES])[:, 1]
    prof["score"] = {"edges": np.unique(np.quantile(scores, np.linspace(0, 1, 11))).tolist(),
                     "sample": rng.choice(scores, size=2000, replace=False).round(6).tolist()}
    return prof


def main():
    t_start = time.perf_counter()
    df = load_raw()
    train, val, test = split(df)
    Xtr, ytr = train[C.FEATURES], train[C.TARGET_COL].to_numpy()
    Xva, yva = val[C.FEATURES], val[C.TARGET_COL].to_numpy()
    neg, pos = (ytr == 0).sum(), (ytr == 1).sum()
    log.info("train %d (churn %.3f) | val %d | test %d", len(train), ytr.mean(), len(val), len(test))

    cand_dir = C.ARTIFACT_DIR / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    results, models = {}, {}

    # 1) Dummy prior: predicts the training churn rate for everyone
    dummy = make_pipeline(DummyClassifier(strategy="prior")).fit(Xtr, ytr)
    models["dummy_prior"] = dummy

    # 2) Logistic regression, class-weighted loss, C tuned by 5-fold CV on TRAIN (PR-AUC)
    lr = RandomizedSearchCV(make_pipeline(LogisticRegression(class_weight="balanced", max_iter=5000)),
                            {"model__C": loguniform(1e-3, 1e2)}, n_iter=20, scoring="average_precision",
                            cv=CV, random_state=C.SEED, n_jobs=1).fit(Xtr, ytr)
    models["logistic_weighted"] = lr.best_estimator_
    log.info("logistic CV PR-AUC %.4f  C=%.4g", lr.best_score_, lr.best_params_["model__C"])

    # 3) LightGBM, class-weighted loss (scale_pos_weight = neg/pos), tuned by 5-fold CV on TRAIN
    space = {"model__n_estimators": randint(100, 600), "model__learning_rate": loguniform(0.01, 0.2),
             "model__num_leaves": randint(4, 32), "model__min_child_samples": randint(10, 100),
             "model__subsample": uniform(0.6, 0.4), "model__subsample_freq": [1],
             "model__colsample_bytree": uniform(0.5, 0.5), "model__reg_lambda": loguniform(1e-3, 10)}
    base = LGBMClassifier(scale_pos_weight=neg / pos, random_state=C.SEED, n_jobs=1, verbose=-1)
    gbm = RandomizedSearchCV(make_pipeline(base), space, n_iter=40, scoring="average_precision", cv=CV,
                             random_state=C.SEED, n_jobs=1).fit(Xtr, ytr)
    models["lightgbm_weighted"] = gbm.best_estimator_
    best_gbm_params = {k.replace("model__", ""): v for k, v in gbm.best_params_.items()}
    log.info("lightgbm CV PR-AUC %.4f", gbm.best_score_)

    # 4) Ablation: identical LightGBM WITHOUT class weighting
    unweighted = make_pipeline(LGBMClassifier(**best_gbm_params, random_state=C.SEED, n_jobs=1, verbose=-1)).fit(Xtr, ytr)
    models["lightgbm_unweighted"] = unweighted

    # 5) Calibrated versions: class weighting inflates probabilities, so the served score is
    #    re-mapped with Platt scaling learned from 5-fold out-of-fold predictions on TRAIN
    #    (validation stays untouched). ensemble=False keeps ONE fitted pipeline + one sigmoid:
    #    benchmark.py shows the 5-model ensemble costs ~4x the latency for no measurable gain.
    for name in ["logistic_weighted", "lightgbm_weighted"]:
        cal = CalibratedClassifierCV(models[name], method="sigmoid", cv=CV, ensemble=False).fit(Xtr, ytr)
        models[name + "_calibrated"] = cal

    # ---- Validation comparison (rule baseline has no score, so it is a single operating point)
    mc_val = val["MonthlyCharges"].to_numpy()
    rflag = rule_flag(val)
    results["heuristic_rule"] = {"pr_auc": None, "roc_auc": None,
                                 "f1_macro": f1_score(yva, rflag, average="macro"),
                                 "precision": float(yva[rflag].mean()), "recall": float(rflag[yva == 1].mean()),
                                 "flag_rate": float(rflag.mean()), "threshold": None,
                                 "net_benefit_usd": net_benefit(yva, rflag, mc_val)}
    curves = {}
    for name, m in models.items():
        p = m.predict_proba(Xva)[:, 1]
        thr, nb, grid, values = best_threshold(yva, p, mc_val)
        results[name] = {**metrics(yva, p, thr), "threshold": thr, "net_benefit_usd": nb}
        curves[name] = values
        joblib.dump(m, cand_dir / f"{name}.joblib")
    results["contact_everyone"] = {"net_benefit_usd": net_benefit(yva, np.ones_like(yva, bool), mc_val)}

    # ---- Selection rule (declared before looking at test): the validation set holds only ~374
    #      churners, so PR-AUC differences below ~0.01 there are noise. Rank deployable candidates by
    #      5-fold CV PR-AUC on TRAIN (five estimates over 4,225 rows); keep a candidate only if its
    #      validation PR-AUC is within 0.01 of the best. Validation then sets the threshold.
    deployable = {"logistic_weighted_calibrated": lr.best_score_, "lightgbm_weighted_calibrated": gbm.best_score_}
    best_val = max(results[n]["pr_auc"] for n in deployable)
    eligible = [n for n in deployable if results[n]["pr_auc"] >= best_val - 0.01]
    chosen = max(eligible, key=lambda n: deployable[n])
    final = models[chosen]
    thr = results[chosen]["threshold"]
    log.info("selected %s  val PR-AUC %.4f  threshold %.2f  net benefit $%.0f",
             chosen, results[chosen]["pr_auc"], thr, results[chosen]["net_benefit_usd"])

    # ---- Leakage audit (with the fitted pipeline, to prove train-only statistics)
    fitted_for_audit = models["lightgbm_weighted"]
    audit = leakage_audit(train, val, test, fitted_for_audit)
    save_json(audit, C.REPORT_DIR / "leakage_audit.json")

    # ---- Export artifacts
    C.ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(final, C.MODEL_FILE)
    version = f"churn-{chosen.split('_')[0]}-{datetime.now(timezone.utc):%Y%m%d}"
    metadata = {
        "model_version": version, "model_name": chosen, "threshold": thr,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": C.FEATURES, "data_sha256": C.DATA_SHA256,
        "split": {"strategy": "stratified 60/20/20", "seed": C.SEED, **audit["rows"]},
        "validation_metrics": results[chosen], "fallback": fallback_table(train),
        "business_assumptions": {"offer_cost_usd": C.OFFER_COST, "offer_success_rate": C.OFFER_SUCCESS_RATE,
                                 "value_horizon_months": C.VALUE_HORIZON_MONTHS},
        "lightgbm_best_params": best_gbm_params, "logistic_best_C": lr.best_params_["model__C"],
        "cv_pr_auc": {"logistic_weighted": lr.best_score_, "lightgbm_weighted": gbm.best_score_},
        "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                             "lightgbm": lightgbm.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    save_json(metadata, C.METADATA_FILE)
    save_json(reference_profile(train, final), C.REFERENCE_FILE)
    save_json({"results": results, "threshold_grid": np.round(np.arange(0.05, 0.951, 0.01), 2).tolist(),
               "net_benefit_curves": curves, "chosen": chosen}, C.REPORT_DIR / "validation_results.json")
    save_json({"train": train[C.ID_COL].tolist(), "val": val[C.ID_COL].tolist(), "test": test[C.ID_COL].tolist()},
              C.ARTIFACT_DIR / "split_ids.json")
    log.info("done in %.1fs", time.perf_counter() - t_start)
    print(json.dumps({k: {m: (round(v, 4) if isinstance(v, float) else v) for m, v in r.items()} for k, r in results.items()}, indent=1))


if __name__ == "__main__":
    main()
