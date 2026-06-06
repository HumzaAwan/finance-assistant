from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, timedelta
from typing import Any, Optional

import httpx
from langchain_core.messages import AIMessage, HumanMessage

from deps import get_runtime
from graph.state import AgentState
from utils.llm_utils import safe_invoke_or_none

_log = logging.getLogger("agent.graph.nodes.transactions_node")


def _today_utc_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).date().isoformat()


def _last_user(messages):
    if not messages:
        return ""

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)

    return str(messages[-1].content)


def _compact_context(messages):
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

    # "This week vs last week" triggers BOTH legacy branches → second overwrote first; widen instead.
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

    allowed = {
        "food",
        "transport",
        "utilities",
        "entertainment",
        "health",
        "shopping",
        "income",
    }

    return lowered if lowered in allowed else None


def transactions_node(state: AgentState):
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
        parsed = parsed_any if isinstance(parsed_any, dict) else {}

    except json.JSONDecodeError:
        parsed = {}

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
            _log.info({"event": "tx_default_window", **{"through": today}})

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
    if intent_kind in {"transaction_query", "insight_request"}:
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
        # LLMs sometimes return the literal string "null"/"none" — drop those.
        if value and str(value).lower().strip() not in _NULL_STRINGS:
            params[key_src] = str(value).split("T")[0]

    endpoint = f"{banking_base}/transactions"
    timeout = float(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))

    fetch_note = ""

    def _pull(p: dict[str, Any]) -> dict:
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(endpoint, params=p)
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            _log.warning({"event": "transactions_fetch_failed", "detail": repr(exc)})
            return {"error": repr(exc), "transactions": [], "count": 0}

    body = _pull(dict(params))

    txs = body.get("transactions") or []

    narrowed = bool(params.get("start_date") or params.get("end_date") or params.get("category"))

    if (
        intent_kind in {"transaction_query", "insight_request"}
        and len(txs) == 0
        and narrowed
        and "error" not in body
    ):
        rebound = {"user_id": params["user_id"], "limit": min(250, max(limit, 180))}
        body2 = _pull(rebound)
        if len(body2.get("transactions") or []) > 0:
            body = body2
            fetch_note = "widened_retry_no_date_category_filters_after_empty_hit"
            params = rebound

    payload = {
        "transactions": body.get("transactions", []) or [],
        "count": body.get("count", len(body.get("transactions") or [])),
        "request_params": dict(params),
        "fetch_note": fetch_note,
        **({"upstream_error": body["error"]} if body.get("error") else {}),
    }

    # Fetch budget targets for intents that may need spending comparisons.
    if intent_kind in {"insight_request", "financial_advice"}:
        budget_endpoint = f"{banking_base}/budgets/{params['user_id']}"
        try:
            with httpx.Client(timeout=timeout) as client:
                br = client.get(budget_endpoint)
                if br.status_code == 200:
                    payload["budgets"] = br.json().get("categories", [])
                    _log.info({"event": "budgets_fetched", "count": len(payload["budgets"])})
        except Exception as exc:
            _log.debug({"event": "budgets_fetch_skipped", "detail": repr(exc)})

    _log.info({"event": "transactions_fetched", "transactions": payload["count"], "filters": dict(params)})

    return {"transaction_data": payload}
