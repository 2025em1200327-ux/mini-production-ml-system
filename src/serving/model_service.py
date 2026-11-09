"""Model loading and scoring, decoupled from the web layer (used by the API, the batch CLI and tests).

Artifacts are loaded once at start-up. If the model artifact is missing or corrupt, or scoring raises
or returns non-finite values, the service degrades to the heuristic rule whose churn rates were
measured on the training data and stored in model_metadata.json - it never returns a silent wrong
answer, and every degraded prediction is labelled `degraded=True`.
"""
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURES, METADATA_FILE, MODEL_FILE

log = logging.getLogger("model_service")
FALLBACK_VERSION = "fallback-rule-v1"


class ModelService:
    def __init__(self, model_path: Path | str | None = None, metadata_path: Path | str | None = None):
        self.model_path = Path(model_path or os.getenv("MODEL_PATH", MODEL_FILE))
        self.metadata_path = Path(metadata_path or os.getenv("METADATA_PATH", METADATA_FILE))
        self.metadata = json.loads(self.metadata_path.read_text())    # small, required: threshold + fallback table
        self.threshold = float(self.metadata["threshold"])
        self.fallback = self.metadata["fallback"]
        self.model, self.load_error = None, None
        try:
            self.model = joblib.load(self.model_path)
            self.model.predict_proba(self._warmup_frame())              # fail fast at start-up, not on first request
            log.info("model loaded: %s", self.metadata["model_version"])
        except Exception as e:                                          # noqa: BLE001 - any load failure degrades
            self.model, self.load_error = None, f"{type(e).__name__}: {e}"
            log.error("model unavailable, serving fallback rule: %s", self.load_error)

    @property
    def model_version(self) -> str:
        return self.metadata["model_version"] if self.model is not None else FALLBACK_VERSION

    @staticmethod
    def _warmup_frame() -> pd.DataFrame:
        row = {"gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 1,
               "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "No",
               "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
               "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
               "PaymentMethod": "Electronic check", "MonthlyCharges": 50.0, "TotalCharges": 50.0}
        return pd.DataFrame([row])[FEATURES]

    def fallback_scores(self, df: pd.DataFrame) -> np.ndarray:
        rule = (df["Contract"] == "Month-to-month") & (df["tenure"] < 12)
        return np.where(rule, self.fallback["p_if_rule"], self.fallback["p_otherwise"]).astype(float)

    def score(self, df: pd.DataFrame) -> tuple[np.ndarray, bool, str]:
        """Return (probabilities, degraded, version). Never raises for a schema-valid frame."""
        X = df[FEATURES]
        if self.model is not None:
            try:
                p = self.model.predict_proba(X)[:, 1]
                if np.all(np.isfinite(p)):
                    return p, False, self.metadata["model_version"]
                log.error("non-finite scores from model; using fallback")
            except Exception as e:                                      # noqa: BLE001
                log.exception("model scoring failed; using fallback: %s", e)
        return self.fallback_scores(X), True, FALLBACK_VERSION
