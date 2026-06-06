from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage

from deps import get_runtime
from graph.state import AgentState
from utils.llm_utils import safe_invoke_or_none

_log = logging.getLogger("agent.graph.nodes.response_node")


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


def _latest_user(messages):

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


def response_node(state: AgentState):
    llm = get_runtime().llm_chat

    # Use the snapshot captured in agent_app.py before graph invocation.
    # We must NOT call memory.get_history() here — messages are only written
    # to Redis after GRAPH.invoke returns, so any direct read gives a view
    # that is missing the current user turn.
    snapshot = state.get("memory_snapshot") or []
    memory_block = "\n".join(f"{row['role'].upper()}: {row['content']}" for row in snapshot)

    question = _latest_user(list(state.get("messages", []) or []))
    intent = state.get("intent", "general")

    rag_lines = ""

    rag_context = state.get("rag_context") or []

    if rag_context:
        rag_lines = "\n".join(f"- [{c['source']} | score={c['score']:.4f}] {c['content']}" for c in rag_context)

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
                "and confirm the sync user_id matches seeded data."
            )

            finale = _squash_single_char_alpha_lines(_strip_leaked_intent_echo(finale))

            return {"final_response": finale}

        if len(txs) == 0:

            extra = ""

            if note:
                extra = f"\n(Internal note after retry: {note})"

            finale = (
                "No transactions matched this request.\n\n"
                f"Filters sent to `/transactions`: {json.dumps(filt)}\n\n"
                "Verify the Streamlit \"User ID\" matches your mock ledger (often `user_001`) and that dates fall "
                f"within the seeded period.{extra}"
            ).strip()

            finale = _squash_single_char_alpha_lines(_strip_leaked_intent_echo(finale))

            return {"final_response": finale}

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
        "If INSIGHT JSON reads as the literal \"none\", you must not cite dollar totals—explain that aggregates were unavailable for this reply."
    )

    human_payload = (
        "AUTHORITATIVE_NUMBERS_RULE: Use INSIGHT JSON (including calendar_week_comparison) for aggregates. "
        "TRANSACTION PREVIEW is truncated (≤35 rows) — never treat it as exhaustive or as the full week's ledger.\n"
        f"classified_intent={intent}\n"
        f"LATEST QUESTION:\n{question}\n\n"
        f"MEMORY (last 5 turns):\n{memory_block or '[empty]'}\n\n"
        f"INSIGHT JSON:\n{insights_text}\n\n"
        f"TRANSACTION PREVIEW JSON (truncated):\n{txn_preview}\n\n"
        f"RAG CONTEXT:\n{rag_lines or '[none]'}\n"
    )

    # Streaming mode: skip LLM call; return assembled context for the caller to stream.
    if state.get("streaming_mode"):
        return {"streaming_context": {"system": system_prompt, "human": human_payload}}

    assistant = safe_invoke_or_none(llm, [("system", system_prompt), ("human", human_payload)])
    if assistant is not None:
        finale = assistant.content.strip()
    else:
        finale = _offline_summary(intent, insights, txs)

    finale = _squash_single_char_alpha_lines(finale)
    finale = _strip_leaked_intent_echo(finale)

    return {"final_response": finale}
