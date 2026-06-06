from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage
from prometheus_client import Counter

from deps import get_runtime
from graph.state import AgentState
from utils.llm_utils import safe_invoke_or_none

_log = logging.getLogger("agent.graph.nodes.response_node")

COMPLIANCE_TRIGGERED = Counter(
    "compliance_triggered_total",
    "FCA regulated-advice guardrail activations",
)

# Phrases that indicate a request for personalised regulated financial advice.
# Bug 6 fix: expanded to cover stock-recommendation phrasing.
_REGULATED_ADVICE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bshould i invest\b", re.IGNORECASE),
    re.compile(r"\bis it worth buying\b", re.IGNORECASE),
    re.compile(r"\brecommend me a fund\b", re.IGNORECASE),
    re.compile(r"\bwhat should i do with my pension\b", re.IGNORECASE),
    re.compile(r"\bshould i buy\b.*\b(stock|share|fund|etf|bond|crypto|bitcoin)\b", re.IGNORECASE),
    re.compile(r"\badvise me (on|about) (invest|portfolio|pension|fund)\b", re.IGNORECASE),
    re.compile(r"\bpick (a|the best) (fund|stock|share|etf)\b", re.IGNORECASE),
    re.compile(r"\bwhich (fund|stock|share) should i\b", re.IGNORECASE),
    # Extended patterns (Bug 6)
    re.compile(r"\brecommend i buy\b", re.IGNORECASE),
    re.compile(r"\bwhich stocks?\b", re.IGNORECASE),
    re.compile(r"\bwhat should i invest\b", re.IGNORECASE),
    re.compile(r"\bbest fund\b", re.IGNORECASE),
    re.compile(r"\bwhich fund\b", re.IGNORECASE),
    re.compile(r"\bshould i put my money in\b", re.IGNORECASE),
    re.compile(r"\bworth investing in\b", re.IGNORECASE),
    re.compile(r"\bgood investment\b", re.IGNORECASE),
    re.compile(r"\bbuy shares?\b", re.IGNORECASE),
    re.compile(r"\bwhere should i invest\b", re.IGNORECASE),
    re.compile(r"\bwhat to invest in\b", re.IGNORECASE),
]

_FCA_FIXED_RESPONSE = (
    "I can provide general financial information and education, but I am not authorised "
    "to give regulated financial advice. For personalised investment recommendations, "
    "please consult an FCA-authorised financial adviser. You can find one at "
    "www.unbiased.co.uk or check the FCA Register at register.fca.org.uk."
)

_GENERAL_ADVICE_DISCLAIMER = (
    "\n\n---\n*This is general financial information only and does not constitute "
    "regulated financial advice.*"
)


def _is_regulated_advice(query: str) -> bool:
    """Return True if the query matches regulated-advice patterns."""
    return any(p.search(query) for p in _REGULATED_ADVICE_PATTERNS)


def _squash_single_char_alpha_lines(text: str) -> str:
    """Some local LLMs insert a newline after every character; repair for readable UI."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        s = lines[i]
        st = s.strip()
        if len(st) == 1 and st.isalpha():
            chars: list[str] = []
            while i < len(lines):
                t = lines[i].strip()
                if len(t) == 1 and t.isalpha():
                    chars.append(t)
                    i += 1
                else:
                    break
            chunk = "".join(chars)
            if out:
                prev = out[-1]
                prev_trim = prev.rstrip()
                sep = ""
                if chunk and prev_trim and not prev_trim[-1].isspace() and chunk[0].isalpha():
                    if prev_trim[-1] not in "-—–":
                        sep = " "
                out[-1] = prev + sep + chunk
            else:
                out.append(chunk)
            continue
        out.append(s)
        i += 1
    return "\n".join(out)


def _strip_leaked_intent_echo(text: str) -> str:
    """Drop lines that mirror hidden prompt metadata (models sometimes paste them verbatim)."""
    kept: list[str] = []
    for line in text.split("\n"):
        probe = line.strip().lstrip("*").strip().lstrip("*").strip()
        if probe.lower().startswith("classified_intent="):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _latest_user(messages) -> str:
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _offline_summary(intent: str, insights: dict | None, txs: list) -> str:
    if not txs:
        return (
            "The chat model (Ollama) is unavailable. No transactions were returned — "
            "check BANKING_API_URL and mock API logs, ensure Ollama is running."
        )

    if intent not in {"transaction_query", "insight_request", "financial_advice"} or not insights:
        return (
            f"Loaded {len(txs)} transactions, but Ollama is unavailable for a fuller answer. "
            "Install/start Ollama and pull llama3.2."
        )

    ins = insights
    parts = sorted((ins.get("by_category") or {}).items())
    breakdown = ", ".join(f"{k}: ${round(float(v), 2)}" for k, v in parts) if parts else "none"

    return (
        f"Summary (offline model): spending total ~ ${ins.get('total_spent')}, largest category \"{ins.get('top_category', '')}\". "
        f"Breakdown: {breakdown}. Based on {len(txs)} fetched transactions."
    )


def _format_health_score(health_score: dict) -> str:
    """Render the financial health score as a readable summary for response_node."""
    score = health_score.get("overall_score", 0)
    grade = health_score.get("grade", "Unknown")
    components = health_score.get("component_scores", {})
    improvement = health_score.get("top_improvement", {})

    lines = [f"**Your Financial Health Score: {score}/100 — {grade}**\n"]

    component_labels = {
        "savings_rate": "Savings Rate (25 pts)",
        "debt_to_income": "Debt-to-Income Ratio (20 pts)",
        "emergency_fund": "Emergency Fund Coverage (25 pts)",
        "budget_adherence": "Budget Adherence (20 pts)",
        "spending_stability": "Spending Stability (10 pts)",
    }

    for key, label in component_labels.items():
        comp = components.get(key, {})
        lines.append(f"- **{label}**: {comp.get('score', 0)}/{comp.get('max', 0)} — {comp.get('explanation', '')}")

    if improvement:
        lines.append(
            f"\n**Top Improvement — {improvement.get('component', '').replace('_', ' ').title()}**: "
            f"{improvement.get('action', '')}"
        )

    return "\n".join(lines)


def _format_anomalies(anomalies: list[dict]) -> str:
    """Render anomaly detection results as a structured user-facing report."""
    if not anomalies:
        return "No suspicious transactions detected in your recent history."

    lines = [f"**{len(anomalies)} suspicious transaction(s) detected:**\n"]
    for flag in anomalies[:20]:  # cap display at 20
        merchant = flag.get("merchant", "Unknown merchant")
        amount = abs(flag.get("amount", 0))
        date = flag.get("date", "")
        rule = flag.get("rule_triggered", "").replace("_", " ").title()
        confidence = flag.get("confidence", "").upper()
        reason = flag.get("reason", "")
        lines.append(f"- [{confidence}] **{rule}** · £{amount:.2f} at {merchant} on {date}: {reason}")

    return "\n".join(lines)


async def response_node(state: AgentState) -> dict:
    llm = get_runtime().llm_chat

    snapshot = state.get("memory_snapshot") or []
    memory_block = "\n".join(f"{row['role'].upper()}: {row['content']}" for row in snapshot)

    question = _latest_user(list(state.get("messages", []) or []))
    intent = state.get("intent", "general")

    # ── FCA Compliance Guardrail ───────────────────────────────────────────────
    if _is_regulated_advice(question):
        COMPLIANCE_TRIGGERED.inc()
        _log.warning({
            "event": "compliance_guardrail_triggered",
            "compliance_triggered": True,
            "intent": intent,
            "query_snippet": question[:80],
        })
        # Return ONLY the fixed message — no LLM call, no disclaimer appended.
        return {
            "final_response": _FCA_FIXED_RESPONSE,
            "compliance_triggered": True,
        }

    # ── Unclear intent: return a single clarifying question ──────────────────
    if intent == "unclear_intent":
        _log.info({
            "event": "unclear_intent_response",
            "unclear_intent": True,
            "message_preview": question[:80],
        })
        # Use LLM with a constrained prompt to ask ONE clarifying question.
        clarify_prompt = (
            "The user's message was unclear. Ask them one specific clarifying question "
            "about their finances. Do not generate financial data or advice. "
            "Respond in one sentence only."
        )
        assistant = safe_invoke_or_none(llm, [("system", clarify_prompt), ("human", question or "[unclear]")])
        if assistant is not None and assistant.content.strip():
            clarification = assistant.content.strip().split("\n")[0]  # one sentence max
        else:
            clarification = (
                "I did not understand that. Could you ask me something about "
                "your transactions, budget, or financial health?"
            )
        clarification = _squash_single_char_alpha_lines(_strip_leaked_intent_echo(clarification))
        return {"final_response": clarification, "compliance_triggered": False}

    # ── Financial health intent: format score directly ────────────────────────
    if intent == "financial_health":
        health_score = state.get("health_score")
        if health_score:
            finale = _format_health_score(health_score)
            finale = _squash_single_char_alpha_lines(_strip_leaked_intent_echo(finale))
            return {"final_response": finale, "compliance_triggered": False}

    # ── Anomaly check intent: format flags directly ───────────────────────────
    if intent == "anomaly_check":
        anomalies = state.get("anomalies") or []
        finale = _format_anomalies(anomalies)
        finale = _squash_single_char_alpha_lines(_strip_leaked_intent_echo(finale))
        return {"final_response": finale, "compliance_triggered": False}

    # ── Standard path ─────────────────────────────────────────────────────────
    rag_context = state.get("rag_context") or []
    rag_lines = ""
    if rag_context:
        rag_parts = []
        for c in rag_context:
            reranker_suffix = f" | reranker={c['reranker_score']:.4f}" if "reranker_score" in c else ""
            rag_parts.append(f"- [{c['source']} | score={c['score']:.4f}{reranker_suffix}] {c['content']}")
        rag_lines = "\n".join(rag_parts)

    insights = state.get("insights")
    insights_text = json.dumps(insights, indent=2) if insights else "none"

    bundle = state.get("transaction_data") or {}
    txs = bundle.get("transactions") or []
    sampled = txs[-35:] if len(txs) > 35 else list(txs)
    txn_preview = json.dumps(sampled, indent=2) if sampled else "none"

    if intent in {"transaction_query", "insight_request", "financial_advice"}:
        upstream = bundle.get("upstream_error")
        note = bundle.get("fetch_note") or ""
        filt = bundle.get("request_params", {})

        if upstream:
            finale = (
                "The banking API did not respond, so no transaction rows were loaded. Technical detail:\n\n"
                f"{upstream}\n\n"
                "Check BANKING_API_URL in .env matches the mock service (typically http://127.0.0.1:8001) "
                "and confirm the user_id matches seeded data."
            )
            finale = _squash_single_char_alpha_lines(_strip_leaked_intent_echo(finale))
            return {"final_response": finale, "compliance_triggered": False}

        if len(txs) == 0:
            extra = f"\n(Internal note after retry: {note})" if note else ""
            finale = (
                "No transactions matched this request.\n\n"
                f"Filters sent to `/transactions`: {json.dumps(filt)}\n\n"
                "Verify the Streamlit \"User ID\" matches your mock ledger (often `user_001`) and that dates fall "
                f"within the seeded period.{extra}"
            ).strip()
            finale = _squash_single_char_alpha_lines(_strip_leaked_intent_echo(finale))
            return {"final_response": finale, "compliance_triggered": False}

    system_prompt = (
        "You are an accurate personal finance copilot connected to transactional JSON, "
        "derived insights, and curated knowledge snippets. Lead with factual numbers when present, "
        "cite categories, acknowledge uncertainty rather than hallucinating unsupported totals, "
        "and weave practical guidance when knowledge snippets arrive. "
        "When insights JSON includes calendar_week_comparison, use current_week_partial_spend and "
        "prior_calendar_week_spend for this week vs last week; do not substitute biggest_transaction "
        "or total_spent for weekly totals. Respect scope_note fields. "
        "Write normal flowing sentences and paragraphs. "
        "Never insert a line break between individual letters; debits are stored as negative numbers in JSON, which is expected, not an error. "
        "Do not paste internal metadata lines (e.g. anything starting with classified_intent=) or raw insight JSON unless the user explicitly asks for technical detail. "
        "If INSIGHT JSON reads as the literal \"none\", you must not cite dollar totals—explain that aggregates were unavailable for this reply. "
        "UK monetary amounts use GBP (£), not USD ($), where applicable."
    )

    # Bug 2 fix: user question is the FIRST line so the LLM cannot lose it
    # inside JSON context. Directive is the LAST line before generation.
    human_payload = (
        f"User question: {question}\n\n"
        "AUTHORITATIVE_NUMBERS_RULE: Use INSIGHT JSON (including calendar_week_comparison) for aggregates. "
        "TRANSACTION PREVIEW is truncated (≤35 rows) — never treat it as exhaustive or as the full week's ledger.\n"
        f"classified_intent={intent}\n\n"
        f"MEMORY (last 5 turns):\n{memory_block or '[empty]'}\n\n"
        f"INSIGHT JSON:\n{insights_text}\n\n"
        f"TRANSACTION PREVIEW JSON (truncated):\n{txn_preview}\n\n"
        f"RAG CONTEXT:\n{rag_lines or '[none]'}\n\n"
        "Answer the user's question directly and specifically. Do not summarise the data unless asked."
    )

    if state.get("streaming_mode"):
        return {
            "streaming_context": {"system": system_prompt, "human": human_payload},
            "compliance_triggered": False,
        }

    assistant = safe_invoke_or_none(llm, [("system", system_prompt), ("human", human_payload)])
    if assistant is not None:
        finale = assistant.content.strip()
    else:
        finale = _offline_summary(intent, insights, txs)

    finale = _squash_single_char_alpha_lines(finale)
    finale = _strip_leaked_intent_echo(finale)

    # Append disclaimer for all non-regulated financial advice responses.
    if intent == "financial_advice":
        finale = finale + _GENERAL_ADVICE_DISCLAIMER

    return {"final_response": finale, "compliance_triggered": False}
