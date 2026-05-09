from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from local_env import apply_local_dev_url_overrides  # noqa: E402

apply_local_dev_url_overrides()

from deps import get_runtime  # noqa: E402  # intentional after env bootstrap
from graph.agent_graph import GRAPH  # noqa: E402


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level_name, logging.INFO))


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    configure_logging()
    get_runtime()
    logging.getLogger("agent.agent_app").info({"event": "service_ready"})
    yield


app = FastAPI(title="Finance LangGraph Agent", lifespan=lifespan)


class ChatPayload(BaseModel):
    message: str
    session_id: str
    user_id: str
    route_hint: str | None = None


@app.post("/chat")
async def chat_endpoint(body: ChatPayload):
    log = logging.getLogger("agent.agent_app.chat")
    memory = get_runtime().memory

    try:
        history = memory.get_history(body.session_id, 5)
        msgs: list = []

        for turn in history:
            role = turn.get("role")
            text = str(turn.get("content", ""))
            if role == "user":
                msgs.append(HumanMessage(content=text))
            elif role == "assistant":
                msgs.append(AIMessage(content=text))

        msgs.append(HumanMessage(content=body.message))

        state = {
            "messages": msgs,
            "user_id": body.user_id,
            "session_id": body.session_id,
            "intent": "",
            "transaction_data": None,
            "insights": None,
            "rag_context": None,
            "final_response": "",
            "route_hint": body.route_hint,
        }

        result = await asyncio.to_thread(GRAPH.invoke, state)
        reply_text = str(result.get("final_response", "")).strip()
        intent_label = str(result.get("intent", ""))

        memory.add_message(body.session_id, "user", body.message)
        memory.add_message(body.session_id, "assistant", reply_text)

        log.info({"event": "chat_complete", "intent": intent_label, "chars": len(reply_text)})

        return {"response": reply_text, "intent": intent_label, "session_id": body.session_id}

    except HTTPException:
        raise

    except Exception:
        log.exception({"event": "chat_failed"})
        raise HTTPException(
            status_code=503,
            detail="Agent invocation failed — see server logs for the traceback.",
        ) from None


@app.get("/chat/history/{session_id}")
async def history_snapshot(session_id: str):
    return {"session_id": session_id, "history": get_runtime().memory.get_full_history(session_id)}


@app.delete("/chat/history/{session_id}")
async def history_delete(session_id: str):
    get_runtime().memory.clear_session(session_id)
    return {"session_id": session_id, **{"status": "cleared"}}


@app.get("/health")
async def health_endpoint():
    return {"status": "ok", "service": "agent"}
