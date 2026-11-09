"""Single source of truth for paths, feature lists, split settings and business-cost assumptions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

# ---- Data source (public, no login needed) -------------------------------------------------
DATA_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
            "master/data/Telco-Customer-Churn.csv")
DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
RAW_FILE = DATA_DIR / "telco_churn.csv"

ID_COL = "customerID"
TARGET_COL = "Churn"          # "Yes" / "No" in the raw file -> 1 / 0

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_NUMERIC = ["SeniorCitizen"]   # already 0/1
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + BINARY_NUMERIC + CATEGORICAL_FEATURES

# ---- Split: stratified 60 / 20 / 20 (no timestamps exist, see report section 2) -------------
SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20               # share of the FULL data

# ---- Business-cost assumptions (stated, not measured; varied in the sensitivity analysis) ---
OFFER_COST = 50.0             # cost of one retention offer, per flagged customer (USD)
OFFER_SUCCESS_RATE = 0.30     # share of contacted true churners who stay because of the offer
VALUE_HORIZON_MONTHS = 12     # a saved customer is worth 12 x MonthlyCharges

MODEL_FILE = ARTIFACT_DIR / "churn_model.joblib"
METADATA_FILE = ARTIFACT_DIR / "model_metadata.json"
REFERENCE_FILE = ARTIFACT_DIR / "reference_profile.json"
