import logging
import re

from langchain_core.messages import HumanMessage

from deps import get_runtime
from graph.state import AgentState
from utils.llm_utils import safe_invoke_or_none

_log = logging.getLogger("agent.graph.nodes.intent_router")

LABELS = {
    "transaction_query",
    "insight_request",
    "financial_advice",
    "general",
}


def _intent_from_heuristics(raw: str) -> str:
    t = raw.lower()

    if any(k in t for k in ["advice", "how do i ", "budget", "save money", "emergency fund", "invest "]):
        return "financial_advice"

    if any(
        k in t
        for k in (
            "summarize",
            "summary",
            "cash flow",
            "overview",
            "trend",
            "compare",
            "versus",
            "break down",
            "breakdown",
            "breaking down",
            "this month",
            "that month",
            "last month",
            "top spending",
            "by category",
            "dig deeper",
            "dive deeper",
        )
    ):
        return "insight_request"

    if any(
        k in t
        for k in [
            "spend",
            "spent",
            "spending",
            "transaction",
            "debit",
            "purchase",
            "categories",
            "how much",
        ]
    ):
        return "transaction_query"

    return "general"


def _mentions_user_ledger(raw: str) -> bool:
    """Signals the user expects account / transaction-grounded facts, not chit-chat."""

    t = raw.lower()

    cues = (
        "spending",
        "spent",
        "expenses",
        "expense",
        "category",
        "categories",
        "transaction",
        "transactions",
        "bills",
        "balance",
        "balances",
        "merchant",
        "payroll",
        "deposit",
    )
    if not any(c in t for c in cues):
        return False

    if any(x in t for x in ["how do i ", "how can i "]) and "spent" not in t and "spending" not in t:
        if not any(z in t for z in ["transaction", "category", "categories"]):
            return False

    phrases = (
        "my ",
        "did i ",
        "have i ",
        "what did i ",
        "how much did i",
        "how much have i",
        "compare ",
        "comparison",
        " vs ",
        " versus ",
        "this week vs",
        "last week vs",
        "breakdown",
        "breaking down",
        "dive deeper",
        "dig deeper",
    )

    topish = ("top " in t and ("spending" in t or "categor" in t)) or ("top categories" in t)

    prefixed = False
    for p in ("show me ", "tell me ", "give me "):
        if t.startswith(p):
            prefixed = True
            break

    return any(ph in t for ph in phrases) or topish or prefixed


SYSTEM_PROMPT = (
    "You are an intent classifier. Labels: "
    "transaction_query (balances, totals, filtering by merchant/category/date), "
    "insight_request (trends, breakdowns, month/week comparisons, 'top categories'), "
    "financial_advice (how-to, budgeting tips unrelated to fetching their ledger), "
    "general (greetings/meta). "
    "Choose the strongest match. Return ONLY the label, lowercase, nothing else."
)


def _last_user_turn(messages):
    if not messages:
        return ""

    candidate = messages[-1]
    if isinstance(candidate, HumanMessage):
        return str(candidate.content).strip()

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content).strip()

    return str(messages[-1].content)


def intent_router(state: AgentState):
    llm = get_runtime().llm_chat
    raw_text = _last_user_turn(list(state.get("messages", []) or []))

    if not raw_text:
        _log.warning({"event": "intent_missing_message", "intent": "general"})
        return {"intent": "general"}

    classifier = safe_invoke_or_none(llm, [("system", SYSTEM_PROMPT), ("human", raw_text.strip())])
    if classifier is not None:
        label_source = classifier.content
    else:
        exc = RuntimeError("LLM unavailable after retries")
        guessed = _intent_from_heuristics(raw_text)
        normalized = guessed if guessed in LABELS else "general"
        if normalized == "general" and _mentions_user_ledger(raw_text):
            normalized = "insight_request"
            _log.info({"event": "intent_ledger_boost", "intent": normalized, "mode": "heuristic_fallback"})
        _log.warning({"event": "intent_llm_failed", "fallback": guessed, "detail": repr(exc)})
        _log.info({"event": "intent_classified", "intent": normalized, "mode": "heuristic_fallback"})
        return {"intent": normalized}

    label = label_source.strip().lower()
    label = re.sub(r"[^a-z_]", "", label.splitlines()[0] if label else "")
    parts = label.split()
    normalized = parts[0] if parts else "general"

    if normalized not in LABELS:
        normalized = "general"

    guess = _intent_from_heuristics(raw_text)
    if normalized == "general" and guess != "general":
        _log.info({"event": "intent_heuristic_upgrade", "from": "general", "to": guess})
        normalized = guess

    if normalized == "general" and _mentions_user_ledger(raw_text):
        normalized = "insight_request"
        _log.info({"event": "intent_ledger_boost", "intent": normalized})

    _log.info({"event": "intent_classified", "intent": normalized})

    return {"intent": normalized}
