from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _agent_api_base(raw: str) -> str:
    """Use 127.0.0.1 when .env still has Docker Compose hostnames (agent, mock-api, etc.)."""

    s = raw.strip().rstrip("/") or "http://127.0.0.1:8000"
    parsed = urlparse(s if "://" in s else f"http://{s}")
    host = (parsed.hostname or "").lower()
    if host == "agent":
        port = parsed.port or 8000
        return f"http://127.0.0.1:{port}"
    return s


BASE_URL = _agent_api_base(os.getenv("AGENT_API_URL", "http://127.0.0.1:8000"))

st.set_page_config(page_title="Finance Assistant", layout="wide")

STARTER_PROMPTS: tuple[str, ...] = (
    "How much did I spend last week?",
    "What are my top spending categories this month?",
    "Give me advice on budgeting",
    "Compare my spending this week vs last week",
    "How do I build an emergency fund?",
)


def ellipsis_join() -> str:
    return "." + "." + "."


def shorten_session_display(session_id: str) -> str:
    mark = ellipsis_join()
    if len(session_id) <= len(mark) + 8:
        return session_id
    return session_id[:8] + mark + session_id[-4:]


def seed_state() -> None:
    if "msgs" not in st.session_state:
        st.session_state.msgs = []
    if "sid" not in st.session_state:
        st.session_state.sid = str(uuid.uuid4())
    if "uid_field" not in st.session_state:
        st.session_state.uid_field = os.getenv("DEFAULT_USER_ID", "user_001")


seed_state()


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
        with st.spinner("Thinking..."):
            data = agent_exchange(clipped)
        st.session_state.msgs.append(
            {
                "role": "assistant",
                "content": str(data.get("response", "")).strip(),
                "intent": str(data.get("intent", "")).strip(),
            }
        )
    except httpx.HTTPError as exc:
        st.session_state.msgs.append(
            {
                "role": "assistant",
                "content": f"The agent API request failed ({type(exc).__name__}: {exc}).",
                "intent": "",
            }
        )
    except Exception as exc:
        st.session_state.msgs.append(
            {
                "role": "assistant",
                "content": f"Something went wrong while contacting the assistant ({type(exc).__name__}: {exc}).",
                "intent": "",
            }
        )
    st.rerun()


def sidebar() -> None:
    st.sidebar.markdown("## 💰 Finance Assistant")
    st.sidebar.text_input("User ID", key="uid_field")
    stub = shorten_session_display(st.session_state.sid)
    st.sidebar.caption(f"Session: `{stub}`")
    if st.sidebar.button("🗑 New Conversation", type="primary"):
        st.session_state.msgs = []
        st.session_state.sid = str(uuid.uuid4())
        st.rerun()
    st.sidebar.divider()
    st.sidebar.caption(
        "About · Local demo UI for the LangGraph finance agent · "
        "messages use session memory on the agent side."
    )


sidebar()


def render_chat(messages: list[dict[str, str]]) -> None:
    if not messages:
        st.markdown("**Welcome.** Pick a starter or type your own question below.")
        cols = st.columns(5)
        for col, starter in zip(cols, STARTER_PROMPTS, strict=True):
            if col.button(starter):
                ship_turn(starter)
        return

    for turn in messages:
        role = turn["role"]
        with st.chat_message(role):
            if role == "assistant":
                intent = turn.get("intent", "")
                if intent:
                    st.caption("Intent • " + intent)
                st.markdown(turn.get("content", ""))
            else:
                st.markdown(turn.get("content", ""))


render_chat(st.session_state.msgs)


chat_prompt = st.chat_input("Ask about spending, budgeting, savings, …")


if chat_prompt:

    ship_turn(chat_prompt)
