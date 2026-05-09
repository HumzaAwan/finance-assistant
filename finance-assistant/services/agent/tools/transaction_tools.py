from __future__ import annotations

import json
import logging
import os
from typing import Any, Mapping, MutableMapping

import httpx

_log = logging.getLogger("agent.tools.transaction_tools")


def banking_base_url() -> str:
    return os.environ["BANKING_API_URL"].rstrip("/")


def fetch_transactions(params: Mapping[str, Any], timeout: float | None = None) -> MutableMapping[str, Any]:
    timeout = timeout or float(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))
    merged = dict(params)
    endpoint = f"{banking_base_url()}/transactions"
    _log.info("%s", json.dumps({"event": "txn_tool_fetch", "endpoint": endpoint, "params_keys": sorted(merged)}))
    with httpx.Client(timeout=timeout) as client:
        response = client.get(endpoint, params=merged)
    response.raise_for_status()
    return response.json()
