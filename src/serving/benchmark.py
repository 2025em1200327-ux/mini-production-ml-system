"""Empirical latency / throughput profiling.

Run:  python -m src.serving.benchmark            (starts its own uvicorn server on port 8765)
Measures:
  1. model-only scoring latency (in-process, one row) - deployed model vs a 5-model calibrated ensemble
  2. HTTP /predict latency, sequential (1,000 requests after 50 warm-up)
  3. HTTP /predict under concurrency (8 in flight, 2,000 requests)
  4. HTTP /predict/batch throughput (100 records per request)
  5. degraded mode: /predict latency when the model artifact is unavailable (fallback rule)
  6. batch CLI throughput (records/sec, validation included) on a 100,000-row file
Writes reports/benchmark.json and reports/figures/latency.png
"""
import asyncio
import json
import os
import platform
import socket
import subprocess
import sys
import time

import httpx
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.data_pipeline import load_raw, split

PORT = 8765


def pct(ms):
    a = np.asarray(ms)
    return {"n": len(a), "p50_ms": round(float(np.percentile(a, 50)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2),
            "p99_ms": round(float(np.percentile(a, 99)), 2), "max_ms": round(float(a.max()), 2),
            "mean_ms": round(float(a.mean()), 2)}


def payloads(df):
    out = []
    for r in df.to_dict(orient="records"):
        p = {k: r[k] for k in C.FEATURES}
        p["TotalCharges"] = None if pd.isna(pd.to_numeric(p["TotalCharges"], errors="coerce")) else float(p["TotalCharges"])
        p["customer_id"] = r[C.ID_COL]
        out.append(p)
    return out


def start_server(env_extra=None):
    env = {**os.environ, "LOG_LEVEL": "WARNING", **(env_extra or {})}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.serving.app:app", "--port", str(PORT),
                             "--log-level", "warning", "--no-access-log"], env=env, cwd=C.ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("server did not start")


def sequential(client, data, n, warmup=50):
    for i in range(warmup):
        client.post("/predict", json=data[i % len(data)])
    ms, errors = [], 0
    for i in range(n):
        t = time.perf_counter()
        r = client.post("/predict", json=data[i % len(data)])
        ms.append((time.perf_counter() - t) * 1000)
        errors += r.status_code != 200
    return ms, errors


async def concurrent(data, n, concurrency):
    sem, ms, errors = asyncio.Semaphore(concurrency), [], 0
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}", timeout=30) as client:
        async def one(i):
            nonlocal errors
            async with sem:
                t = time.perf_counter()
                r = await client.post("/predict", json=data[i % len(data)])
                ms.append((time.perf_counter() - t) * 1000)
                errors += r.status_code != 200
        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(n)))
        wall = time.perf_counter() - t0
    return ms, errors, wall


def main():
    _, val, test = split(load_raw())
    data = payloads(test)
    res = {"environment": {"python": platform.python_version(), "platform": platform.platform(),
                           "cpu_count": os.cpu_count(), "processor": platform.processor() or "unknown",
                           "server": "uvicorn, 1 worker"}}

    # 1) Model-only latency: deployed (single calibrated pipeline) vs 5-model calibrated ensemble
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold
    deployed = joblib.load(C.MODEL_FILE)
    train, _, _ = split(load_raw())
    base = joblib.load(C.ARTIFACT_DIR / "candidates" / "lightgbm_weighted.joblib")
    ensemble = CalibratedClassifierCV(base, method="sigmoid", cv=StratifiedKFold(5, shuffle=True, random_state=C.SEED),
                                      ensemble=True).fit(train[C.FEATURES], train[C.TARGET_COL])
    row = test[C.FEATURES].iloc[[0]]
    res["model_only"] = {}
    for name, m in [("deployed_single_calibrated", deployed), ("ensemble_5_calibrated", ensemble)]:
        for _ in range(30):
            m.predict_proba(row)
        ms = []
        for _ in range(500):
            t = time.perf_counter(); m.predict_proba(row); ms.append((time.perf_counter() - t) * 1000)
        res["model_only"][name] = pct(ms)
        X = pd.concat([test[C.FEATURES]] * 10, ignore_index=True)
        t = time.perf_counter(); m.predict_proba(X); res["model_only"][name]["vectorised_rows_per_sec"] = round(len(X) / (time.perf_counter() - t))

    # 2-4) HTTP benchmarks against a real uvicorn process
    proc = start_server()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=30) as client:
            ms, err = sequential(client, data, 1000)
            res["http_predict_sequential"] = {**pct(ms), "errors": err, "requests_per_sec": round(1000 / (sum(ms) / 1000), 1)}
            seq_ms = ms
            bms, berr = [], 0
            for i in range(100):
                chunk = [data[(i * 100 + j) % len(data)] for j in range(100)]
                t = time.perf_counter(); r = client.post("/predict/batch", json={"records": chunk})
                bms.append((time.perf_counter() - t) * 1000); berr += r.status_code != 200
            res["http_batch_100"] = {**pct(bms), "errors": berr, "records_per_sec": round(100 * 100 / (sum(bms) / 1000))}
        ms, err, wall = asyncio.run(concurrent(data, 2000, 8))
        res["http_predict_concurrency_8"] = {**pct(ms), "errors": err, "throughput_rps": round(2000 / wall, 1)}
        conc_ms = ms
    finally:
        proc.terminate(); proc.wait()

    # 5) Degraded mode: model artifact unavailable -> fallback rule
    proc = start_server({"MODEL_PATH": "/nonexistent/model.joblib"})
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=30) as client:
            health = client.get("/health").json()
            ms, err = sequential(client, data, 500)
            sample = client.post("/predict", json=data[0]).json()
        res["http_predict_fallback"] = {**pct(ms), "errors": err, "health": health, "sample_response": sample}
    finally:
        proc.terminate(); proc.wait()

    # 6) Batch CLI throughput on 100,000 rows (resampled test rows, unique ids)
    from src.serving.predict_batch import score_file
    big = test.sample(100_000, replace=True, random_state=C.SEED).reset_index(drop=True)
    big[C.ID_COL] = [f"BENCH-{i:06d}" for i in range(len(big))]
    big.drop(columns=[C.TARGET_COL]).to_csv("/tmp/bench_100k.csv", index=False)
    res["batch_cli_100k"] = score_file("/tmp/bench_100k.csv", "/tmp/bench_100k_scored.csv", "/tmp/bench_100k_rejects.csv")

    (C.REPORT_DIR / "benchmark.json").write_text(json.dumps(res, indent=2))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bins = np.linspace(0, np.percentile(conc_ms, 99.5), 60)
    ax.hist(seq_ms, bins=bins, alpha=.7, label="sequential (1 in flight)", color="#2F6DB5")
    ax.hist(conc_ms, bins=bins, alpha=.6, label="8 in flight", color="#C2571A")
    for v, c in [(res["http_predict_sequential"]["p95_ms"], "#2F6DB5"), (res["http_predict_concurrency_8"]["p95_ms"], "#C2571A")]:
        ax.axvline(v, c=c, ls="--", lw=1)
    ax.set_xlabel("End-to-end /predict latency (ms)"); ax.set_ylabel("Requests"); ax.legend(fontsize=8)
    ax.set_title("API latency distribution (dashed = p95)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "latency.png", dpi=180)
    print(json.dumps({k: v for k, v in res.items() if k != "environment"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
