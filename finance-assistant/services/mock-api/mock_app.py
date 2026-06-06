from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from routes.accounts import router as accounts_router
from routes.budgets import router as budgets_router
from routes.transactions import router as transactions_router

# ── Prometheus metrics ────────────────────────────────────────────────────────
MOCK_REQUESTS = Counter("mock_api_requests_total", "Total mock API requests", ["endpoint"])
MOCK_LATENCY = Histogram("mock_api_duration_seconds", "Mock API request latency", ["endpoint"])


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level_name, logging.INFO))


configure_logging()

log = logging.getLogger("mock_api.mock_app")


def _evt(payload: dict) -> str:
    return json.dumps(payload, default=str)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    log.info("%s", _evt({"event": "startup"}))
    yield
    log.info("%s", _evt({"event": "shutdown"}))


app = FastAPI(title="Finance Mock Banking API", lifespan=lifespan)
app.include_router(transactions_router)
app.include_router(accounts_router)
app.include_router(budgets_router)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - t0
    label = request.url.path.split("/")[1] if request.url.path.count("/") >= 1 else "root"
    MOCK_REQUESTS.labels(endpoint=label).inc()
    MOCK_LATENCY.labels(endpoint=label).observe(elapsed)
    return response


@app.get("/health")
async def health():
    log.info("%s", _evt({"event": "health_check", "service": "mock-api"}))
    return {"status": "ok", "service": "mock-api"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
