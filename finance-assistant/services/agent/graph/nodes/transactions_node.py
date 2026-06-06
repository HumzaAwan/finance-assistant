from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, timedelta
from typing import Any, Literal, Optional

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from prometheus_client import Counter
from pydantic import BaseModel, Field, field_validator

from deps import get_runtime
from graph.state import AgentState
from utils.llm_utils import safe_invoke_or_none

_log = logging.getLogger("agent.graph.nodes.transactions_node")

# ── Prometheus counters for structured output reliability ──────────────────────
STRUCTURED_OUTPUT_SUCCESS = Counter(
    "structured_output_success_total",
    "LLM structured output (TxQueryParams) parsed successfully",
)
STRUCTURED_OUTPUT_FALLBACK = Counter(
    "structured_output_fallback_total",
    "LLM structured output failed; fell back to regex extraction",
)

CategoryLiteral = Literal["food", "transport", "utilities", "entertainment", "health", "shopping", "income"]


class TxQueryParams(BaseModel):
    """Structured query parameters extracted from the user's natural-language request."""

    start_date: Optional[date] = Field(None, description="Start date for transaction filter (YYYY-MM-DD)")
    end_date: Optional[date] = Field(None, description="End date for transaction filter (YYYY-MM-DD)")
    category: Optional[CategoryLiteral] = Field(None, description="Transaction category to filter by")
    limit: int = Field(default=20, ge=1, le=250, description="Maximum number of transactions to return")
    user_id: Optional[str] = Field(None, description="User ID override; defaults to state user_id")

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def coerce_null_strings(cls, v: Any) -> Any:
        if isinstance(v, str) and v.lower().strip() in {"null", "none", "undefined", "n/a", ""}:
            return None
        return v


def _today_utc_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).date().isoformat()


def _last_user(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return str(messages[-1].content) if messages else ""


def _compact_context(messages) -> str:
    lines = []
    snippet = messages[-12:] if len(messages) >= 12 else messages
    for m in snippet:
        if isinstance(m, HumanMessage):
            lines.append("user: " + str(m.content))
        elif isinstance(m, AIMessage):
            lines.append("assistant: " + str(m.content))
    return "\n".join(lines[-8:])


def _wants_week_over_week_compare(question: str) -> bool:
    q = question.lower()
    if "week" not in q:
        return False
    if "compare" in q or "comparison" in q:
        return True
    if " vs " in q or " versus " in q:
        return True
    this_markers = ("this week", "current week")
    last_markers = ("last week", "prior week", "previous week")
    return any(m in q for m in this_markers) and any(m in q for m in last_markers)


def _fallback_date_window(raw_question: str, today_iso: str) -> dict[str, Any]:
    """Lightweight phrase → date-range when LLM extraction is unavailable."""
    q = raw_question.lower()
    d = date.fromisoformat(today_iso.split("T")[0])
    params: dict[str, Any] = {}
    monday_this = d - timedelta(days=d.weekday())

    if _wants_week_over_week_compare(raw_question):
        span_start = monday_this - timedelta(days=7)
        params["start_date"] = span_start.isoformat()
        params["end_date"] = d.isoformat()
        return params

    if any(p in q for p in ("last week", "previous week", "prior week")):
        s = monday_this - timedelta(days=7)
        params["start_date"] = s.isoformat()
        params["end_date"] = (s + timedelta(days=6)).isoformat()

    if "this week" in q or "current week" in q:
        s = monday_this
        params["start_date"] = s.isoformat()
        params["end_date"] = d.isoformat()

    return params


def _normalize_category(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    lowered = str(value).lower().strip()
    if lowered in {"null", "none"}:
        return None
    allowed = {"food", "transport", "utilities", "entertainment", "health", "shopping", "income"}
    return lowered if lowered in allowed else None


def _extract_params_structured(llm, system_prompt: str, transcript: str, raw_question: str) -> dict[str, Any]:
    """Primary path: use with_structured_output(TxQueryParams) for reliable JSON extraction."""
    try:
        structured_llm = llm.with_structured_output(TxQueryParams, method="json_mode")
        human_msg = f"CONVERSATION SNIPPET:\n{transcript}\n\nQUESTION:\n{raw_question}"
        result: TxQueryParams = structured_llm.invoke([("system", system_prompt), ("human", human_msg)])

        if not isinstance(result, TxQueryParams):
            raise ValueError(f"Unexpected return type: {type(result)}")

        STRUCTURED_OUTPUT_SUCCESS.inc()
        _log.info({"event": "tx_structured_output_success"})

        parsed: dict[str, Any] = {}
        if result.start_date:
            parsed["start_date"] = result.start_date.isoformat()
        if result.end_date:
            parsed["end_date"] = result.end_date.isoformat()
        if result.category:
            parsed["category"] = result.category
        parsed["limit"] = result.limit
        return parsed

    except Exception as exc:
        _log.warning({"event": "tx_structured_output_failed", "detail": repr(exc), "fallback": "regex"})
        STRUCTURED_OUTPUT_FALLBACK.inc()
        return _extract_params_fallback(llm, system_prompt, transcript, raw_question)


def _extract_params_fallback(llm, system_prompt: str, transcript: str, raw_question: str) -> dict[str, Any]:
    """Fallback path: regex-clean raw LLM text → json.loads()."""
    extractor = safe_invoke_or_none(
        llm,
        [("system", system_prompt), ("human", f"CONVERSATION SNIPPET:\n{transcript}\n\nQUESTION:\n{raw_question}")],
    )
    if extractor is not None:
        cleaned = extractor.content.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "").strip()
    else:
        _log.warning({"event": "tx_extract_llm_failed", "fallback": "empty_params"})
        cleaned = "{}"

    try:
        parsed_any = json.loads(cleaned)
        return parsed_any if isinstance(parsed_any, dict) else {}
    except json.JSONDecodeError:
        return {}


async def _pull_async(endpoint: str, params: dict[str, Any], timeout: float) -> dict:
    """Async HTTP GET using httpx.AsyncClient."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        _log.warning({"event": "transactions_fetch_failed", "detail": repr(exc)})
        return {"error": repr(exc), "transactions": [], "count": 0}


async def transactions_node(state: AgentState) -> dict:
    banking_base = os.environ["BANKING_API_URL"].rstrip("/")
    intent_kind = str(state.get("intent") or "")

    llm = get_runtime().llm_chat
    messages = list(state.get("messages", []) or [])
    transcript = _compact_context(messages)
    raw_question = _last_user(messages)
    today = _today_utc_iso()

    system_prompt = (
        'Given banking questions, reply with STRICT JSON ONLY: {"start_date":"YYYY-MM-DD or null",'
        '"end_date":"YYYY-MM-DD or null","category":"food|transport|utilities|entertainment'
        '|health|shopping|income|null","limit":integer}. '
        f"Interpret relative expressions using today UTC {today}. Dates must remain realistic."
    )

    # Primary: structured output; fallback: regex
    parsed = _extract_params_structured(llm, system_prompt, transcript, raw_question)

    fb_window = _fallback_date_window(raw_question, today)
    compare_weeks = _wants_week_over_week_compare(raw_question)
    if compare_weeks:
        parsed.update(dict(fb_window))
    else:
        for key, val in fb_window.items():
            cur = parsed.get(key)
            if cur in (None, "", "null"):
                parsed[key] = val

    if intent_kind in {"transaction_query", "insight_request"} and not compare_weeks:
        if not parsed.get("start_date") and not parsed.get("end_date"):
            d_ref = date.fromisoformat(today.split("T")[0])
            parsed["start_date"] = (d_ref - timedelta(days=89)).isoformat()
            parsed["end_date"] = today
            _log.info({"event": "tx_default_window", "through": today})

    if compare_weeks:
        cur = parsed.get("limit")
        if cur in (None, "", "null"):
            parsed["limit"] = 200
        else:
            try:
                parsed["limit"] = max(int(cur), 150)
            except (TypeError, ValueError):
                parsed["limit"] = 200

    limit_raw = parsed.get("limit") or 50
    try:
        lim = int(limit_raw)
    except (TypeError, ValueError):
        lim = 50
    # anomaly_check needs a large window for reliable z-score baselines (Bug 5B fix).
    if intent_kind in {"transaction_query", "insight_request", "anomaly_check"}:
        lim = min(250, max(lim, 120))

    limit = max(1, min(lim, 250))

    params: dict[str, Any] = {
        "user_id": state.get("user_id", os.getenv("DEFAULT_USER_ID", "user_001")),
        "limit": limit,
    }

    normalized_category = _normalize_category(parsed.get("category"))
    if normalized_category:
        params["category"] = normalized_category

    _NULL_STRINGS = {"null", "none", "undefined", "n/a", ""}
    for key_src in ("start_date", "end_date"):
        value = parsed.get(key_src)
        if value and str(value).lower().strip() not in _NULL_STRINGS:
            params[key_src] = str(value).split("T")[0]

    endpoint = f"{banking_base}/transactions"
    timeout = float(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))

    body = await _pull_async(endpoint, dict(params), timeout)
    txs = body.get("transactions") or []
    narrowed = bool(params.get("start_date") or params.get("end_date") or params.get("category"))
    fetch_note = ""

    if (
        intent_kind in {"transaction_query", "insight_request"}
        and len(txs) == 0
        and narrowed
        and "error" not in body
    ):
        rebound = {"user_id": params["user_id"], "limit": min(250, max(limit, 180))}
        body2 = await _pull_async(endpoint, rebound, timeout)
        if len(body2.get("transactions") or []) > 0:
            body = body2
            fetch_note = "widened_retry_no_date_category_filters_after_empty_hit"
            params = rebound

    payload: dict[str, Any] = {
        "transactions": body.get("transactions", []) or [],
        "count": body.get("count", len(body.get("transactions") or [])),
        "request_params": dict(params),
        "fetch_note": fetch_note,
        **({"upstream_error": body["error"]} if body.get("error") else {}),
    }

    # Fetch budget targets and account data for intents that may need them.
    if intent_kind in {"insight_request", "financial_advice", "financial_health"}:
        budget_endpoint = f"{banking_base}/budgets/{params['user_id']}"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                br = await client.get(budget_endpoint)
                if br.status_code == 200:
                    payload["budgets"] = br.json().get("categories", [])
                    _log.info({"event": "budgets_fetched", "count": len(payload["budgets"])})
        except Exception as exc:
            _log.debug({"event": "budgets_fetch_skipped", "detail": repr(exc)})

    if intent_kind in {"financial_health", "anomaly_check"}:
        accounts_endpoint = f"{banking_base}/accounts"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                ar = await client.get(accounts_endpoint, params={"user_id": params["user_id"]})
                if ar.status_code == 200:
                    payload["accounts"] = ar.json()
                    _log.info({"event": "accounts_fetched"})
        except Exception as exc:
            _log.debug({"event": "accounts_fetch_skipped", "detail": repr(exc)})

    _log.info({"event": "transactions_fetched", "transactions": payload["count"], "filters": dict(params)})
    return {"transaction_data": payload}
