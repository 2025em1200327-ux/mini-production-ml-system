PY ?= python

.PHONY: all setup data train evaluate drift diagram test serve benchmark batch

all: data train evaluate drift diagram test

setup:            ## create a virtual environment with locked dependencies
	$(PY) -m venv .venv && .venv/bin/pip install -r requirements.lock

data:             ## download + checksum-verify the dataset
	$(PY) -m src.fetch_data

train:            ## baselines, tuning, calibration, threshold, artifacts
	$(PY) -m src.train

evaluate:         ## one-time test-set evaluation, figures
	$(PY) -m src.evaluate

drift:            ## drift monitor demo (stable vs simulated shock)
	$(PY) -m src.drift_demo

diagram:
	$(PY) -m src.architecture_diagram

test:
	$(PY) -m pytest -q

serve:            ## online API on http://localhost:8000 (docs at /docs)
	$(PY) -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

benchmark:        ## latency percentiles + throughput (starts its own server)
	$(PY) -m src.serving.benchmark

batch:            ## score the sample file with the batch CLI
	$(PY) -m src.serving.predict_batch --input data/sample_new_customers.csv --output reports/sample_scored.csv
