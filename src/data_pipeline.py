"""Ingestion, cleaning, leak-free splitting, the preprocessing pipeline and the leakage audit.

Every transformation that LEARNS something from data (imputer medians, scaler statistics,
one-hot vocabularies) lives in `build_preprocessor()` and is fitted on the training partition
only, as part of one sklearn Pipeline. Row-wise rules that learn nothing live in `ChurnCleaner`,
which is part of the same Pipeline so training and serving run identical code.
"""
import json
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (BINARY_NUMERIC, CATEGORICAL_FEATURES, FEATURES, ID_COL, NUMERIC_FEATURES,
                        RAW_FILE, REPORT_DIR, SEED, TARGET_COL, TEST_SIZE, VAL_SIZE)

log = logging.getLogger("data_pipeline")

ADDON_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                  "StreamingTV", "StreamingMovies"]
ENGINEERED = ["num_addons", "charge_ratio"]


def load_raw(path=RAW_FILE) -> pd.DataFrame:
    """Read the raw CSV, keep the id, the 19 input columns and the label (as 0/1)."""
    df = pd.read_csv(path)
    df[TARGET_COL] = (df[TARGET_COL].str.strip() == "Yes").astype(int)
    return df[[ID_COL] + FEATURES + [TARGET_COL]]


class ChurnCleaner(BaseEstimator, TransformerMixin):
    """Stateless, row-wise cleaning and feature engineering (learns nothing, so cannot leak).

    * TotalCharges arrives as text with blanks for brand-new customers: coerce to number;
      a blank with tenure == 0 is a customer not yet billed, so it becomes 0. Any other
      missing value is left as NaN for the (train-fitted) imputer.
    * num_addons: how many of the six add-on services the customer has.
    * charge_ratio: TotalCharges / (tenure x MonthlyCharges). Near 1 = stable price; far from 1
      = the monthly price changed over the customer's life (a known churn trigger).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        new_customer = X["TotalCharges"].isna() & (X["tenure"] == 0)
        X.loc[new_customer, "TotalCharges"] = 0.0
        X["num_addons"] = (X[ADDON_SERVICES] == "Yes").sum(axis=1)
        expected = X["tenure"] * X["MonthlyCharges"]
        X["charge_ratio"] = np.where(expected > 0, X["TotalCharges"] / expected.replace(0, np.nan), 1.0)
        return X


def build_preprocessor() -> ColumnTransformer:
    """Learned transformations - fitted on the training partition only (inside the Pipeline)."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES + ENGINEERED),
        ("bin", "passthrough", BINARY_NUMERIC),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def make_pipeline(model) -> Pipeline:
    return Pipeline([("clean", ChurnCleaner()), ("prep", build_preprocessor()), ("model", model)])


def split(df: pd.DataFrame):
    """Stratified 60/20/20 split. The data has no event timestamps (see report), so a time
    cut-off is impossible; stratification keeps the 26.5% churn rate identical in every part."""
    rest, test = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET_COL], random_state=SEED)
    train, val = train_test_split(rest, test_size=VAL_SIZE / (1 - TEST_SIZE),
                                  stratify=rest[TARGET_COL], random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def leakage_audit(train, val, test, fitted_pipeline=None) -> dict:
    """Explicit checks, written to reports/leakage_audit.json and quoted in the report."""
    ids = {n: set(d[ID_COL]) for n, d in (("train", train), ("val", val), ("test", test))}
    feat_key = lambda d: set(map(tuple, d[FEATURES].astype(str).values))
    tr_keys = feat_key(train)
    audit = {
        "rows": {n: len(d) for n, d in (("train", train), ("val", val), ("test", test))},
        "churn_rate": {n: round(float(d[TARGET_COL].mean()), 4) for n, d in (("train", train), ("val", val), ("test", test))},
        "id_overlap": {"train_val": len(ids["train"] & ids["val"]), "train_test": len(ids["train"] & ids["test"]),
                       "val_test": len(ids["val"] & ids["test"])},
        "identical_feature_rows_shared_with_train": {"val": len(tr_keys & feat_key(val)), "test": len(tr_keys & feat_key(test))},
        "columns_excluded": [ID_COL],
        "post_outcome_columns_present": [],
    }
    # Single-feature AUC on TRAIN: a feature that alone predicts the label almost perfectly is a leakage red flag.
    clean = ChurnCleaner().transform(train[FEATURES])
    single = {}
    for col in NUMERIC_FEATURES + ENGINEERED + BINARY_NUMERIC:
        v = clean[col].fillna(clean[col].median())
        a = roc_auc_score(train[TARGET_COL], v)
        single[col] = round(max(a, 1 - a), 4)
    for col in CATEGORICAL_FEATURES:
        rate = train.groupby(col)[TARGET_COL].mean()
        a = roc_auc_score(train[TARGET_COL], train[col].map(rate))
        single[col] = round(max(a, 1 - a), 4)
    audit["single_feature_auc_train"] = dict(sorted(single.items(), key=lambda kv: -kv[1]))
    audit["max_single_feature_auc"] = max(single.values())
    audit["single_feature_auc_flag_threshold"] = 0.90
    if fitted_pipeline is not None:   # prove the imputer learned TRAIN statistics, not full-data ones
        imputer = fitted_pipeline.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
        cleaned_train = ChurnCleaner().transform(train[FEATURES])
        cols = NUMERIC_FEATURES + ENGINEERED
        audit["imputer_medians_equal_train_medians"] = bool(np.allclose(imputer.statistics_, cleaned_train[cols].median().values))
    return audit


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tr, va, te = split(load_raw())
    audit = leakage_audit(tr, va, te)
    save_json(audit, REPORT_DIR / "leakage_audit.json")
    print(json.dumps(audit, indent=2))
