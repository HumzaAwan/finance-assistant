from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from routes.transactions import router as transactions_router


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


@app.get("/health")
async def health():
    log.info("%s", _evt({"event": "health_check", **{"service": "mock-api"}}))
    return {"status": "ok", "service": "mock-api"}
