# Mini Production ML System — Customer Churn Scoring

End-to-end churn prediction service for a telecom subscriber base: leak-free data pipeline, class-weighted
LightGBM with calibrated probabilities and a **cost-based decision threshold**, a validated **FastAPI** service
with a heuristic **fallback**, a **batch CLI**, empirical **latency profiling**, and a **PSI/KS drift monitor**.

ML Model Engineering — Graded Assignment 1 (BITS Pilani WILP) · Karandeep Saluja · 1200327

| Test-set result (1,409 unseen customers) | Value |
|---|---|
| PR-AUC (primary metric; random = 0.265) | **0.666** (95% CI 0.612–0.713) |
| ROC-AUC | 0.846 |
| Recall / precision at the deployed threshold 0.20 | 85.8% / 46.7% |
| Net benefit vs contacting nobody (stated cost assumptions) | **$51.6k** (rule of thumb: $27.7k) |
| `/predict` latency, 1 CPU core, sequential | p50 13.2 ms · p95 20.2 ms · p99 21.8 ms |
| Batch CLI throughput (100k rows, validation included) | ~24,000 records/s |

## Quickstart

Requires **Python 3.12** (tested on 3.12.3), Linux or macOS. Windows: run the same `python -m ...` commands.

```bash
git clone https://github.com/2025em1200327-ux/mini-production-ml-system.git
cd mini-production-ml-system
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock          # exact, fully locked environment (requirements.txt = direct pins)

python -m src.fetch_data                  # download + SHA-256 verify the public dataset (~1 MB)
python -m src.train                       # ~1 min: baselines, tuning, calibration, threshold, artifacts
python -m src.evaluate                    # test-set metrics + figures in reports/
python -m pytest -q                       # 36 tests
```

The trained artifacts are committed in `artifacts/`, so the API and batch CLI work right after
`pip install` — retraining is only needed to reproduce the numbers (`make all` runs every stage).

### Online API

```bash
uvicorn src.serving.app:app --port 8000          # interactive docs: http://localhost:8000/docs
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "customer_id": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL",
  "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
  "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85}'
```

Response: `churn_probability`, `churn_flag`, `decision` (`offer_retention` / `no_action`), `threshold`,
`model_version`, `degraded`. Errors: **422** field-level validation errors, **413** batch above 1,000 records,
**500** unexpected (logged with the `X-Request-ID`). If the model cannot load or fails at scoring, the service
keeps answering from the fallback rule and marks responses `"degraded": true`; `/health` reports `degraded`.
Set `PREDICTION_LOG=predictions.jsonl` to log inputs and scores for drift monitoring.

### Batch scoring

```bash
python -m src.serving.predict_batch --input data/sample_new_customers.csv --output reports/sample_scored.csv
# -> reports/sample_scored.csv (customerID, probability, flag, decision, model_version)
# -> reports/sample_scored.rejects.csv (rows that failed validation, with the reason); exit code 2 if any
```

### Profiling and monitoring

```bash
python -m src.serving.benchmark      # p50/p95/p99 latency, concurrency, fallback mode, batch throughput
python -m src.monitoring --input data/sample_new_customers.csv   # drift report; exit code 2 on ALERT
python -m src.drift_demo             # stable batches vs a simulated market shock
```

## Repository layout

```
README.md  requirements.txt  requirements.lock  Makefile
data/        sample_new_customers.csv (demo batch with 3 deliberate edge cases); raw CSV is downloaded
artifacts/   churn_model.joblib · model_metadata.json (version, threshold, fallback rule) · reference_profile.json · split_ids.json
reports/     leakage_audit.json · validation_results.json · test_metrics.json · benchmark.json · drift_demo.json · figures/
src/
  config.py            paths, features, split and business-cost assumptions
  fetch_data.py        download + checksum
  data_pipeline.py     loading, row-wise cleaner, train-only preprocessing, stratified split, leakage audit
  train.py             baselines, class-weighted LR and LightGBM, CV tuning, calibration, cost threshold, export
  evaluate.py          one-time test evaluation, bootstrap intervals, sensitivity, figures
  monitoring.py        PSI / KS / unseen-category / score-drift checks (CLI)
  drift_demo.py        monitor demonstration
  architecture_diagram.py
  serving/
    schemas.py         Pydantic contracts (enums, ranges, cross-field rules)
    model_service.py   artifact loading, scoring, fallback
    app.py             FastAPI service
    predict_batch.py   batch CLI
    benchmark.py       latency / throughput profiling
tests/       schemas, API + fallbacks, inference integrity (train/serve skew), leakage, monitoring
```

## Key design decisions

* **Split:** stratified 60/20/20 (seed 42). The dataset is a single snapshot with no event dates, so a time
  cut-off is impossible; the report discusses the snapshot-leakage risk this leaves and how production
  would remove it.
* **No leakage:** every learned transform sits inside one sklearn `Pipeline` fitted on the training split only;
  tests prove the fitted imputer/scaler statistics equal training-split statistics.
* **Imbalance:** class-weighted loss (`scale_pos_weight`, `class_weight="balanced"`), then Platt calibration
  (5-fold, training split) so served probabilities are honest; no oversampling.
* **Decision:** threshold chosen on validation to maximise expected net benefit (offer cost $50, 30% save rate,
  12 months of the customer's own charges) — assumptions live in `src/config.py`.

## Data

IBM Telco Customer Churn sample (7,043 customers, 26.5% churn), public on GitHub:
`IBM/telco-customer-churn-on-icp4d` (`data/Telco-Customer-Churn.csv`). Downloaded and verified by `src/fetch_data.py`.
