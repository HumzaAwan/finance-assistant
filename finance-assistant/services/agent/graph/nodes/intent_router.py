from __future__ import annotations

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
    "financial_health",
    "anomaly_check",
    "unclear_intent",
    "general",
}

SYSTEM_PROMPT = (
    "You are an intent classifier. Labels:\n"
    "  transaction_query — balances, totals, filtering by merchant/category/date\n"
    "  insight_request — trends, breakdowns, month/week comparisons, 'top categories'\n"
    "  financial_advice — how-to, budgeting tips, ISA/pension/tax/savings questions\n"
    "  financial_health — 'my financial health', 'overall score', 'financial situation summary', 'am I on track'\n"
    "  anomaly_check — suspicious transactions, unusual activity, fraud, duplicate charges\n"
    "  unclear_intent — message is incoherent, gibberish, or too vague to classify\n"
    "  general — greetings, meta questions about the assistant\n"
    "Choose the strongest match. Return ONLY the label, lowercase, nothing else."
)

# ── Gibberish / low-signal detection ──────────────────────────────────────────

# A practical set of recognisable English words used to detect gibberish input.
_ENGLISH_VOCAB: frozenset[str] = frozenset({
    # Finance / banking
    "spend", "spending", "spent", "money", "finance", "financial", "budget",
    "income", "expense", "expenses", "transaction", "transactions", "account",
    "accounts", "balance", "payment", "payments", "bank", "credit", "debit",
    "fund", "invest", "investment", "tax", "pension", "debt", "loan",
    "mortgage", "salary", "savings", "save", "advice", "health", "score",
    "total", "amount", "category", "categories", "monthly", "weekly", "annual",
    "interest", "rate", "stock", "stocks", "share", "shares", "buy", "sell",
    "market", "portfolio", "fraud", "suspicious", "unusual", "duplicate",
    "charge", "charges", "isa", "sipp", "lisa", "jisa", "cash", "transfer",
    "deposit", "withdrawal", "overdraft", "statement", "report", "summary",
    "trend", "comparison", "analysis", "history", "recent", "latest", "check",
    # Query verbs
    "list", "show", "get", "find", "give", "tell", "compare", "calculate",
    "analyze", "analyse", "explain", "describe", "summarize", "summarise",
    "look", "see", "view", "search", "review", "track", "monitor", "help",
    # Common English words
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "its",
    "was", "one", "our", "out", "had", "how", "did", "get", "has", "him",
    "his", "about", "which", "what", "when", "where", "who", "why", "from",
    "this", "that", "they", "them", "then", "than", "with", "will", "been",
    "have", "there", "their", "were", "your", "like", "just", "into", "over",
    "make", "know", "take", "time", "good", "well", "much", "more", "also",
    "back", "after", "first", "most", "last", "down", "long", "need", "want",
    "year", "month", "week", "day", "today", "here", "many", "some", "only",
    "come", "could", "would", "should", "these", "those", "same", "high",
    "old", "new", "next", "used", "work", "way", "each", "between", "going",
    "large", "big", "own", "right", "mean", "put", "set", "try", "around",
    "every", "great", "think", "say", "ask", "plan", "even", "real", "true",
    "please", "okay", "yes", "no", "sure", "maybe", "perhaps", "might",
    "actually", "really", "quite", "very", "too", "so", "because", "since",
    "while", "during", "through", "without", "within", "across", "along",
    # Time words
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday",
    # Pronouns / articles
    "i", "my", "me", "we", "he", "she", "it", "a", "an", "is", "be", "do",
    "at", "by", "in", "of", "on", "to", "up", "if", "as", "or",
})


def _real_word_count(text: str) -> int:
    """Count tokens that appear in the English vocabulary (case-insensitive)."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return sum(1 for t in tokens if len(t) >= 2 and t in _ENGLISH_VOCAB)


def _is_gibberish(text: str) -> bool:
    """Return True if the text contains no recognisable English words at all."""
    return _real_word_count(text) == 0


def _is_low_signal(text: str) -> bool:
    """Return True when fewer than 2 recognisable English words are found.

    Catches pure gibberish ("asdfjkl xyz 123") before wasting an LLM call.
    """
    alpha_tokens = re.findall(r"[a-zA-Z]+", text)
    if not alpha_tokens:
        return True
    return _real_word_count(text) < 2


# ── Keyword heuristics ────────────────────────────────────────────────────────

# Strong UK-finance signals that always override an incorrect LLM classification.
_FINANCIAL_ADVICE_STRONG: frozenset[str] = frozenset({
    "isa", "lisa", "jisa", "sipp", "annuity", "ns&i", "premium bond",
    "income tax", "capital gains", "self assessment", "national insurance",
    "credit score", "credit rating", "experian", "equifax", "transunion",
    "open banking", "psd2", "compound interest", "index fund", "etf",
    "diversification", "inflation", "real return", "avalanche", "snowball",
    "debt payoff", "pension",
})

_ANOMALY_STRONG: frozenset[str] = frozenset({
    "suspicious", "fraud", "charged twice", "duplicate charge",
    "strange transaction", "unexpected charge", "unauthorised",
})

_FINANCIAL_HEALTH_STRONG: frozenset[str] = frozenset({
    "financial health", "health score", "financial score",
    "how am i doing financially", "am i on track", "overall finances",
    "financial situation",
})


def _intent_from_heuristics(raw: str) -> str:
    t = raw.lower()

    # financial_health strong signals
    if any(k in t for k in [
        "financial health", "health score", "overall score", "am i on track",
        "financial situation", "financial assessment", "rate my finances",
        "how am i doing financially", "financial score", "overall finances",
    ]):
        return "financial_health"

    # anomaly_check — expanded
    if any(k in t for k in [
        "suspicious", "unusual activity", "fraud", "anomaly", "anomalies",
        "duplicate charge", "strange transaction", "flagged", "alert",
        "charged twice", "unexpected charge", "unauthorised", "unusual",
    ]):
        return "anomaly_check"

    # financial_advice — original + UK finance product terms
    if any(k in t for k in [
        "advice", "how do i ", "budget", "save money", "emergency fund", "invest ",
        "isa", "lisa", "jisa", "sipp", "pension", "annuity", "ns&i", "premium bond",
        "income tax", "capital gains", "self assessment", "national insurance",
        "credit score", "credit rating", "experian", "equifax", "transunion",
        "open banking", "psd2", "avalanche", "snowball", "debt payoff",
        "compound interest", "index fund", "etf", "diversification",
        "inflation", "real return", "tax", "nisa", "isas",
    ]):
        return "financial_advice"

    # insight_request — expanded with compare / overspend phrasings
    if any(
        k in t
        for k in (
            "summarize", "summary", "cash flow", "overview", "trend",
            "compare", "week vs", "this week vs", "last week", "month vs",
            "versus", " vs ", "break down", "breakdown", "breaking down",
            "this month", "that month", "last month", "top spending",
            "by category", "dig deeper", "dive deeper", "overspending",
            "over budget", "which category", "where am i spending",
            "summarise my spending", "spending summary",
        )
    ):
        return "insight_request"

    if any(
        k in t
        for k in ["spend", "spent", "spending", "transaction", "debit", "purchase", "categories", "how much"]
    ):
        return "transaction_query"

    return "general"


def _apply_strong_overrides(normalized: str, raw_text: str) -> str:
    """Apply hard keyword overrides for high-confidence domain signals.

    The LLM can misclassify UK finance product names as 'transaction_query'.
    When we detect a strong domain signal, we trust the heuristic over the LLM.
    """
    t = raw_text.lower()

    # Strong financial_advice signals override any non-financial_advice label
    if normalized not in {"financial_advice"} and any(k in t for k in _FINANCIAL_ADVICE_STRONG):
        _log.info({"event": "intent_strong_override", "from": normalized, "to": "financial_advice"})
        return "financial_advice"

    # Strong anomaly signals
    if normalized not in {"anomaly_check"} and any(k in t for k in _ANOMALY_STRONG):
        _log.info({"event": "intent_strong_override", "from": normalized, "to": "anomaly_check"})
        return "anomaly_check"

    # Strong financial_health signals
    if normalized not in {"financial_health"} and any(k in t for k in _FINANCIAL_HEALTH_STRONG):
        _log.info({"event": "intent_strong_override", "from": normalized, "to": "financial_health"})
        return "financial_health"

    return normalized


def _mentions_user_ledger(raw: str) -> bool:
    """Signals the user expects account / transaction-grounded facts, not chit-chat."""
    t = raw.lower()
    cues = (
        "spending", "spent", "expenses", "expense", "category", "categories",
        "transaction", "transactions", "bills", "balance", "balances",
        "merchant", "payroll", "deposit",
    )
    if not any(c in t for c in cues):
        return False

    if any(x in t for x in ["how do i ", "how can i "]) and "spent" not in t and "spending" not in t:
        if not any(z in t for z in ["transaction", "category", "categories"]):
            return False

    phrases = (
        "my ", "did i ", "have i ", "what did i ", "how much did i", "how much have i",
        "compare ", "comparison", " vs ", " versus ", "this week vs", "last week vs",
        "breakdown", "breaking down", "dive deeper", "dig deeper",
    )
    topish = ("top " in t and ("spending" in t or "categor" in t)) or ("top categories" in t)
    prefixed = any(t.startswith(p) for p in ("show me ", "tell me ", "give me "))
    return any(ph in t for ph in phrases) or topish or prefixed


def _last_user_turn(messages) -> str:
    if not messages:
        return ""
    candidate = messages[-1]
    if isinstance(candidate, HumanMessage):
        return str(candidate.content).strip()
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content).strip()
    return str(messages[-1].content)


async def intent_router(state: AgentState) -> dict:
    llm = get_runtime().llm_chat
    raw_text = _last_user_turn(list(state.get("messages", []) or []))

    if not raw_text:
        _log.warning({"event": "intent_missing_message", "intent": "general"})
        return {"intent": "general"}

    # ── Gibberish / low-signal check (fast path, no LLM needed) ──────────────
    if _is_gibberish(raw_text):
        _log.info({
            "event": "intent_gibberish_detected",
            "intent": "unclear_intent",
            "message_preview": raw_text[:60],
        })
        return {"intent": "unclear_intent"}

    if _is_low_signal(raw_text):
        _log.info({
            "event": "intent_low_signal",
            "intent": "unclear_intent",
            "message_preview": raw_text[:60],
        })
        return {"intent": "unclear_intent"}

    # ── LLM classification ────────────────────────────────────────────────────
    classifier = safe_invoke_or_none(llm, [("system", SYSTEM_PROMPT), ("human", raw_text.strip())])
    if classifier is not None:
        label_source = classifier.content
    else:
        guessed = _intent_from_heuristics(raw_text)
        normalized = guessed if guessed in LABELS else "general"
        normalized = _apply_strong_overrides(normalized, raw_text)
        if normalized == "general" and _mentions_user_ledger(raw_text):
            normalized = "insight_request"
            _log.info({"event": "intent_ledger_boost", "intent": normalized, "mode": "heuristic_fallback"})
        _log.warning({"event": "intent_llm_failed", "fallback": normalized})
        _log.info({"event": "intent_classified", "intent": normalized, "mode": "heuristic_fallback"})
        return {"intent": normalized}

    label = label_source.strip().lower()
    label = re.sub(r"[^a-z_]", "", label.splitlines()[0] if label else "")
    parts = label.split()
    normalized = parts[0] if parts else "general"

    if normalized not in LABELS:
        normalized = "general"

    # Heuristic upgrade from "general" (original behaviour)
    guess = _intent_from_heuristics(raw_text)
    if normalized == "general" and guess != "general":
        _log.info({"event": "intent_heuristic_upgrade", "from": "general", "to": guess})
        normalized = guess

    # Strong-signal overrides regardless of LLM classification (Bug 3 fix)
    normalized = _apply_strong_overrides(normalized, raw_text)

    if normalized == "general" and _mentions_user_ledger(raw_text):
        normalized = "insight_request"
        _log.info({"event": "intent_ledger_boost", "intent": normalized})

    _log.info({"event": "intent_classified", "intent": normalized})
    return {"intent": normalized}
