import json

import numpy as np
from lightgbm import LGBMClassifier

from src import config as C
from src.data_pipeline import ChurnCleaner, load_raw, make_pipeline, split
from src.monitoring import drift_report, psi, simulate_drift


def test_split_disjoint_and_stratified():
    tr, va, te = split(load_raw())
    assert not (set(tr[C.ID_COL]) & set(va[C.ID_COL])) and not (set(tr[C.ID_COL]) & set(te[C.ID_COL]))
    assert abs(tr[C.TARGET_COL].mean() - te[C.TARGET_COL].mean()) < 0.005


def test_preprocessing_learns_from_train_only():
    df = load_raw()
    tr, va, te = split(df)
    pipe = make_pipeline(LGBMClassifier(n_estimators=5, verbose=-1)).fit(tr[C.FEATURES], tr[C.TARGET_COL])
    imputer = pipe.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    scaler = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    cols = C.NUMERIC_FEATURES + ["num_addons", "charge_ratio"]
    train_clean = ChurnCleaner().transform(tr[C.FEATURES])[cols]
    full_clean = ChurnCleaner().transform(df[C.FEATURES])[cols]
    assert np.allclose(imputer.statistics_, train_clean.median().values)
    assert np.allclose(scaler.mean_, train_clean.fillna(train_clean.median()).mean().values)
    assert not np.allclose(scaler.mean_, full_clean.fillna(full_clean.median()).mean().values)   # i.e. NOT full data


def test_cleaner_is_row_wise():
    _, _, te = split(load_raw())
    batch = ChurnCleaner().transform(te[C.FEATURES].head(50))
    single = ChurnCleaner().transform(te[C.FEATURES].iloc[[7]])
    assert batch.iloc[7].equals(single.iloc[0])


def test_psi_zero_for_identical_and_positive_for_shift():
    a = np.array([0.25, 0.25, 0.5])
    assert psi(a, a) < 1e-9 and psi(a, np.array([0.5, 0.25, 0.25])) > 0.1


def test_monitor_quiet_on_test_and_alerts_on_shock():
    _, _, te = split(load_raw())
    ref = json.loads(C.REFERENCE_FILE.read_text())
    assert drift_report(te, ref)["overall"] == "ok"
    assert drift_report(simulate_drift(te), ref)["overall"] == "ALERT"
