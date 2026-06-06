from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# ── URL helpers ───────────────────────────────────────────────────────────────

def _normalise(raw: str, docker_host: str, default_port: int) -> str:
    s = raw.strip().rstrip("/") or f"http://127.0.0.1:{default_port}"
    parsed = urlparse(s if "://" in s else f"http://{s}")
    host = (parsed.hostname or "").lower()
    if host == docker_host:
        port = parsed.port or default_port
        return f"http://127.0.0.1:{port}"
    return s


BASE_URL = _normalise(os.getenv("AGENT_API_URL", "http://127.0.0.1:8000"), "agent", 8000)
MOCK_URL = _normalise(os.getenv("BANKING_API_URL", "http://127.0.0.1:8001"), "mock-api", 8001)

st.set_page_config(page_title="Finance Assistant", layout="wide", page_icon="💰")

STARTER_PROMPTS: tuple[str, ...] = (
    "How much did I spend last week?",
    "What are my top spending categories this month?",
    "Give me advice on budgeting",
    "Compare my spending this week vs last week",
    "How do I build an emergency fund?",
)


# ── Session state ─────────────────────────────────────────────────────────────

def seed_state() -> None:
    if "msgs" not in st.session_state:
        st.session_state.msgs = []
    if "sid" not in st.session_state:
        st.session_state.sid = str(uuid.uuid4())
    if "uid_field" not in st.session_state:
        st.session_state.uid_field = os.getenv("DEFAULT_USER_ID", "user_001")


seed_state()


# ── API helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=10, show_spinner=False)
def fetch_json(url: str, timeout: float = 5.0) -> dict | list | None:
    try:
        r = httpx.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=10, show_spinner=False)
def fetch_text(url: str, timeout: float = 5.0) -> str | None:
    try:
        r = httpx.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception:
        return None


# ── Prometheus parser ─────────────────────────────────────────────────────────

def parse_prometheus(text: str) -> dict[str, list[dict]]:
    """Parse Prometheus text exposition format into {metric_name: [{labels, value}]}."""
    metrics: dict[str, list[dict]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # metric{labels} value [timestamp]
        m = re.match(r'([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}\s+([0-9eE+\-.]+)', line)
        if m:
            name, labels_raw, val_s = m.groups()
            labels: dict[str, str] = {}
            for lm in re.finditer(r'(\w+)="([^"]*)"', labels_raw):
                labels[lm.group(1)] = lm.group(2)
            try:
                metrics.setdefault(name, []).append({"labels": labels, "value": float(val_s)})
            except ValueError:
                pass
            continue
        m2 = re.match(r'([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([0-9eE+\-.]+)', line)
        if m2:
            name, val_s = m2.groups()
            try:
                metrics.setdefault(name, []).append({"labels": {}, "value": float(val_s)})
            except ValueError:
                pass
    return metrics


def metric_total(parsed: dict, name: str) -> float:
    return sum(e["value"] for e in parsed.get(name, []))


def metric_by_label(parsed: dict, name: str, label: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for e in parsed.get(name, []):
        key = e["labels"].get(label, "unknown")
        out[key] = out.get(key, 0.0) + e["value"]
    return out


def latency_p95(parsed: dict, name: str) -> float | None:
    """Estimate p95 from a Prometheus histogram bucket set."""
    buckets = [
        (float(e["labels"]["le"]), e["value"])
        for e in parsed.get(f"{name}_bucket", [])
        if e["labels"].get("le") not in ("+Inf", None)
        and e["labels"].get("le", "").replace(".", "").isdigit()
    ]
    buckets.sort(key=lambda x: x[0])
    if not buckets:
        return None
    total_entries = [e["value"] for e in parsed.get(f"{name}_bucket", []) if e["labels"].get("le") == "+Inf"]
    total = total_entries[0] if total_entries else (buckets[-1][1] if buckets else 0)
    if total == 0:
        return None
    target = total * 0.95
    for le, count in buckets:
        if count >= target:
            return le
    return buckets[-1][0]


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar() -> None:
    with st.sidebar:
        st.markdown("## 💰 Finance Assistant")
        st.text_input("User ID", key="uid_field")
        stub = st.session_state.sid[:8] + "…" + st.session_state.sid[-4:]
        st.caption(f"Session: `{stub}`")
        if st.button("🗑 New Conversation", type="primary", use_container_width=True):
            st.session_state.msgs = []
            st.session_state.sid = str(uuid.uuid4())
            st.rerun()
        st.divider()
        st.caption("Local demo · LangGraph + Chroma + Ollama")


sidebar()


# ── Chat helpers ──────────────────────────────────────────────────────────────

def agent_exchange(prompt: str) -> dict[str, Any]:
    rsp = httpx.post(
        f"{BASE_URL}/chat",
        json={
            "message": prompt,
            "session_id": st.session_state.sid,
            "user_id": st.session_state.uid_field,
        },
        timeout=180.0,
    )
    rsp.raise_for_status()
    return rsp.json()


def ship_turn(prompt: str) -> None:
    clipped = prompt.strip()
    if not clipped:
        return
    st.session_state.msgs.append({"role": "user", "content": clipped})
    try:
        with st.spinner("Thinking…"):
            data = agent_exchange(clipped)
        st.session_state.msgs.append({
            "role": "assistant",
            "content": str(data.get("response", "")).strip(),
            "intent": str(data.get("intent", "")).strip(),
        })
    except httpx.HTTPError as exc:
        st.session_state.msgs.append({
            "role": "assistant",
            "content": f"Agent API request failed ({type(exc).__name__}: {exc}).",
            "intent": "",
        })
    except Exception as exc:
        st.session_state.msgs.append({
            "role": "assistant",
            "content": f"Something went wrong ({type(exc).__name__}: {exc}).",
            "intent": "",
        })
    st.rerun()


# ── Tab layout ────────────────────────────────────────────────────────────────

chat_tab, dash_tab = st.tabs(["💬 Chat", "📊 Live Dashboard"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Chat
# ═══════════════════════════════════════════════════════════════════════════════

with chat_tab:
    messages = st.session_state.msgs

    if not messages:
        st.markdown("**Welcome.** Pick a starter or type your own question below.")
        cols = st.columns(5)
        for col, starter in zip(cols, STARTER_PROMPTS):
            if col.button(starter, use_container_width=True):
                ship_turn(starter)
    else:
        for turn in messages:
            role = turn["role"]
            with st.chat_message(role):
                if role == "assistant":
                    intent = turn.get("intent", "")
                    if intent:
                        st.caption(f"Intent • {intent}")
                    st.markdown(turn.get("content", ""))
                else:
                    st.markdown(turn.get("content", ""))

    chat_prompt = st.chat_input("Ask about spending, budgeting, savings…")
    if chat_prompt:
        ship_turn(chat_prompt)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

with dash_tab:

    # ── Header + refresh ──────────────────────────────────────────────────────
    hdr_col, btn_col = st.columns([5, 1])
    with hdr_col:
        st.markdown("### 📊 Live System Dashboard")
    with btn_col:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    user_id = st.session_state.uid_field or "user_001"

    # ── Fetch all data in parallel (cached 10 s) ──────────────────────────────
    agent_health   = fetch_json(f"{BASE_URL}/health")
    mock_health    = fetch_json(f"{MOCK_URL}/health")
    agent_metrics  = parse_prometheus(fetch_text(f"{BASE_URL}/metrics") or "")
    mock_metrics   = parse_prometheus(fetch_text(f"{MOCK_URL}/metrics") or "")
    accounts_data  = fetch_json(f"{MOCK_URL}/accounts?user_id={user_id}")
    budgets_data   = fetch_json(f"{MOCK_URL}/budgets/{user_id}")
    tx_summary     = fetch_json(f"{MOCK_URL}/transactions/summary?user_id={user_id}&period=monthly")

    st.divider()

    # ── Row 1 — Service health ────────────────────────────────────────────────
    st.markdown("#### 🏥 Service Health")
    h1, h2, h3 = st.columns(3)

    with h1:
        agent_ok = isinstance(agent_health, dict) and agent_health.get("status") == "ok"
        st.metric(
            "Agent API (8000)",
            "🟢 Online" if agent_ok else "🔴 Offline",
            delta="Ollama: " + (agent_health.get("ollama", "unknown") if agent_ok else "—"),
            delta_color="normal" if (agent_ok and agent_health.get("ollama") == "reachable") else "off",
        )

    with h2:
        mock_ok = isinstance(mock_health, dict) and mock_health.get("status") == "ok"
        st.metric("Mock Banking API (8001)", "🟢 Online" if mock_ok else "🔴 Offline")

    with h3:
        total_requests = metric_total(agent_metrics, "agent_chat_requests_total")
        rag_hits = metric_total(agent_metrics, "agent_rag_hits_total")
        rag_rate = f"{rag_hits / total_requests * 100:.0f}%" if total_requests > 0 else "—"
        st.metric("RAG Hit Rate", rag_rate, help="% of chat requests that retrieved at least one RAG chunk")

    st.divider()

    # ── Row 2 — Chat metrics ──────────────────────────────────────────────────
    st.markdown("#### 💬 Chat Metrics")
    m1, m2, m3, m4 = st.columns(4)

    p95 = latency_p95(agent_metrics, "agent_chat_duration_seconds")
    avg_count = metric_total(agent_metrics, "agent_chat_duration_seconds_count") or None
    avg_sum = metric_total(agent_metrics, "agent_chat_duration_seconds_sum") or None
    avg_lat = (avg_sum / avg_count) if (avg_sum and avg_count) else None

    with m1:
        st.metric("Total Requests", int(total_requests) if total_requests else 0)
    with m2:
        st.metric("RAG Hits", int(rag_hits) if rag_hits else 0)
    with m3:
        st.metric("Avg Latency", f"{avg_lat:.1f}s" if avg_lat else "—")
    with m4:
        st.metric("p95 Latency", f"{p95:.1f}s" if p95 else "—")

    # Intent distribution chart
    intent_counts = metric_by_label(agent_metrics, "agent_chat_requests_total", "intent")
    if intent_counts:
        st.markdown("**Requests by Intent**")
        import pandas as pd
        intent_df = pd.DataFrame(
            {"Intent": list(intent_counts.keys()), "Requests": list(intent_counts.values())}
        ).sort_values("Requests", ascending=False)
        st.bar_chart(intent_df.set_index("Intent"), height=200)
    else:
        st.info("No chat requests yet — start a conversation in the Chat tab.")

    st.divider()

    # ── Row 3 — Mock API endpoint usage ──────────────────────────────────────
    st.markdown("#### 🏦 Mock API — Endpoint Usage")
    endpoint_counts = metric_by_label(mock_metrics, "mock_api_requests_total", "endpoint")
    if endpoint_counts:
        import pandas as pd
        ep_df = pd.DataFrame(
            {"Endpoint": list(endpoint_counts.keys()), "Hits": list(endpoint_counts.values())}
        ).sort_values("Hits", ascending=False)
        st.bar_chart(ep_df.set_index("Endpoint"), height=180)
    else:
        st.info("No requests recorded by mock API metrics yet.")

    st.divider()

    # ── Row 4 — Accounts ─────────────────────────────────────────────────────
    st.markdown(f"#### 💳 Accounts — {user_id}")
    if isinstance(accounts_data, list) and accounts_data:
        acc_cols = st.columns(len(accounts_data))
        for col, acc in zip(acc_cols, accounts_data):
            with col:
                balance = acc.get("balance", 0.0)
                acct_type = acc.get("account_type", "account").title()
                st.metric(
                    label=f"{'🏦' if acct_type == 'Checking' else '💰'} {acc.get('name', acct_type)}",
                    value=f"${balance:,.2f}",
                    help=f"Type: {acct_type} · Currency: {acc.get('currency', 'USD')}",
                )
    elif not agent_ok:
        st.warning("Agent API offline — cannot load account data.")
    else:
        st.info("No accounts found. Ensure the mock API is running and seed data is loaded.")

    st.divider()

    # ── Row 5 — Budget vs Actual ──────────────────────────────────────────────
    st.markdown(f"#### 🎯 Budget vs Actual — {user_id} (this month)")

    budget_cats: list[dict] = []
    if isinstance(budgets_data, dict):
        budget_cats = budgets_data.get("categories", [])

    # Pull actuals from transaction summary if available
    actuals: dict[str, float] = {}
    if isinstance(tx_summary, dict):
        by_cat = tx_summary.get("by_category", {})
        for cat, val in (by_cat or {}).items():
            actuals[cat] = abs(float(val))

    if budget_cats:
        import pandas as pd

        rows = []
        for b in budget_cats:
            cat = b.get("category", "")
            limit = float(b.get("monthly_limit", 0))
            actual = actuals.get(cat, 0.0)
            pct = (actual / limit * 100) if limit > 0 else 0.0
            status = "🔴 Over" if pct > 100 else ("🟡 Warning" if pct > 80 else "🟢 OK")
            rows.append({
                "Category": cat.title(),
                "Spent ($)": round(actual, 2),
                "Budget ($)": limit,
                "Used %": round(pct, 1),
                "Status": status,
            })

        budget_df = pd.DataFrame(rows).sort_values("Used %", ascending=False)

        # Bar chart: budget vs actual side by side
        chart_df = budget_df.set_index("Category")[["Spent ($)", "Budget ($)"]]
        st.bar_chart(chart_df, height=250)

        # Summary table
        st.dataframe(
            budget_df[["Category", "Spent ($)", "Budget ($)", "Used %", "Status"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No budget targets configured. Use `PUT /budgets/{user_id}` to set targets.")

    st.divider()

    # ── Row 6 — Raw metrics inspector ────────────────────────────────────────
    with st.expander("🔬 Raw Prometheus metrics", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Agent API `/metrics`**")
            raw_agent = fetch_text(f"{BASE_URL}/metrics")
            if raw_agent:
                st.code(raw_agent[:3000] + ("…" if len(raw_agent) > 3000 else ""), language="text")
            else:
                st.error("Cannot reach agent /metrics")
        with col_b:
            st.markdown("**Mock API `/metrics`**")
            raw_mock = fetch_text(f"{MOCK_URL}/metrics")
            if raw_mock:
                st.code(raw_mock[:3000] + ("…" if len(raw_mock) > 3000 else ""), language="text")
            else:
                st.error("Cannot reach mock API /metrics")

    st.caption(f"Data cached for 10 s · Last loaded at {time.strftime('%H:%M:%S')} · User: {user_id}")
