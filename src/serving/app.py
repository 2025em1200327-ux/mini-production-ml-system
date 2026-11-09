"""FastAPI online scoring service.

Run:  uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
Endpoints: GET /health, GET /model-info, POST /predict, POST /predict/batch
Errors:    422 invalid payload (field-level reasons), 413 batch too large, 500 unexpected (logged with request id)
"""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.serving.model_service import ModelService
from src.serving.schemas import (MAX_BATCH, BatchRequest, BatchResponse, CustomerFeatures, ErrorResponse,
                                 HealthResponse, PredictionResponse)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")
PREDICTION_LOG = os.getenv("PREDICTION_LOG")        # optional JSONL of inputs + scores, feeds drift monitoring


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = ModelService()              # artifacts loaded ONCE, before traffic is accepted
    yield


app = FastAPI(title="Churn scoring service", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Process-Time-ms"] = f"{ms:.2f}"
    log.info(json.dumps({"rid": request.state.request_id, "path": request.url.path, "status": response.status_code,
                         "ms": round(ms, 2)}))
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    detail = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]} for e in exc.errors()]
    too_many = any(e.get("type") == "too_long" and "records" in [str(p) for p in e["loc"]] for e in exc.errors())
    code = 413 if too_many else 422
    body = ErrorResponse(error="batch_too_large" if too_many else "validation_error", detail=detail,
                         request_id=getattr(request.state, "request_id", None))
    return JSONResponse(status_code=code, content=body.model_dump())


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    log.exception("unhandled error rid=%s", getattr(request.state, "request_id", None))
    body = ErrorResponse(error="internal_error", detail="Unexpected server error; it has been logged.",
                         request_id=getattr(request.state, "request_id", None))
    return JSONResponse(status_code=500, content=body.model_dump())


def _to_frame(records: list[CustomerFeatures]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump(exclude={"customer_id"}) for r in records])


def _respond(records, probs, degraded, version, threshold) -> list[PredictionResponse]:
    out = []
    for r, p in zip(records, probs):
        flag = bool(p >= threshold)
        out.append(PredictionResponse(customer_id=r.customer_id, churn_probability=round(float(p), 6),
                                      churn_flag=flag, decision="offer_retention" if flag else "no_action",
                                      threshold=threshold, model_version=version, degraded=degraded))
    return out


def _log_predictions(records, probs, version):
    if not PREDICTION_LOG:
        return
    with open(PREDICTION_LOG, "a") as f:
        for r, p in zip(records, probs):
            f.write(json.dumps({**r.model_dump(), "score": float(p), "model_version": version, "ts": time.time()}) + "\n")


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    s = request.app.state.service
    return HealthResponse(status="ok" if s.model is not None else "degraded", model_loaded=s.model is not None,
                          model_version=s.model_version, fallback_available=True)


@app.get("/model-info")
def model_info(request: Request):
    s = request.app.state.service
    m = s.metadata
    return {"model_version": s.model_version, "threshold": s.threshold, "features": m["features"],
            "trained_at_utc": m["trained_at_utc"], "validation_metrics": m["validation_metrics"],
            "fallback_rule": s.fallback, "load_error": s.load_error, "max_batch": MAX_BATCH}


@app.post("/predict", response_model=PredictionResponse, responses={422: {"model": ErrorResponse}})
def predict(record: CustomerFeatures, request: Request):
    s = request.app.state.service
    probs, degraded, version = s.score(_to_frame([record]))
    _log_predictions([record], probs, version)
    return _respond([record], probs, degraded, version, s.threshold)[0]


@app.post("/predict/batch", response_model=BatchResponse,
          responses={413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
def predict_batch(batch: BatchRequest, request: Request):
    s = request.app.state.service
    probs, degraded, version = s.score(_to_frame(batch.records))
    _log_predictions(batch.records, probs, version)
    preds = _respond(batch.records, probs, degraded, version, s.threshold)
    return BatchResponse(n=len(preds), model_version=version, degraded=degraded, predictions=preds)
