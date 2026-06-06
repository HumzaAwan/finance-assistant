from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import Response

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from local_env import apply_local_dev_url_overrides  # noqa: E402

apply_local_dev_url_overrides()

from deps import check_ollama_health, get_runtime  # noqa: E402
from graph.agent_graph import GRAPH  # noqa: E402

# ── Prometheus metrics ────────────────────────────────────────────────────────
CHAT_REQUESTS = Counter(
    "agent_chat_requests_total", "Total chat requests", ["intent"]
)
CHAT_LATENCY = Histogram(
    "agent_chat_duration_seconds", "End-to-end chat request latency"
)
RAG_HITS = Counter(
    "agent_rag_hits_total", "RAG queries returning at least one chunk above threshold"
)

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level_name, logging.INFO))


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    configure_logging()
    get_runtime()
    check_ollama_health()
    logging.getLogger("agent.agent_app").info({"event": "service_ready"})
    yield


app = FastAPI(title="Finance LangGraph Agent", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Request / response schemas ────────────────────────────────────────────────

class ChatPayload(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-\.]+$")
    user_id: str = Field(..., min_length=1, max_length=64)
    route_hint: str | None = Field(None, max_length=256)


# ── Shared state-builder ──────────────────────────────────────────────────────

def _build_state(body: ChatPayload, history: list, streaming_mode: bool = False) -> dict:
    msgs: list = []
    for turn in history:
        role = turn.get("role")
        text = str(turn.get("content", ""))
        if role == "user":
            msgs.append(HumanMessage(content=text))
        elif role == "assistant":
            msgs.append(AIMessage(content=text))
    msgs.append(HumanMessage(content=body.message))

    return {
        "messages": msgs,
        "user_id": body.user_id,
        "session_id": body.session_id,
        "intent": "",
        "transaction_data": None,
        "insights": None,
        "rag_context": None,
        "final_response": "",
        "route_hint": body.route_hint,
        "memory_snapshot": history,
        "streaming_mode": streaming_mode,
        "streaming_context": None,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat")
@limiter.limit("30/minute")
async def chat_endpoint(request: Request, body: ChatPayload):  # noqa: ARG001
    log = logging.getLogger("agent.agent_app.chat")
    memory = get_runtime().memory

    try:
        t0 = time.perf_counter()
        history = memory.get_history(body.session_id, 5)
        state = _build_state(body, history)

        result = await asyncio.to_thread(GRAPH.invoke, state)
        reply_text = str(result.get("final_response", "")).strip()
        intent_label = str(result.get("intent", ""))
        rag_ctx = result.get("rag_context") or []

        memory.add_message(body.session_id, "user", body.message)
        memory.add_message(body.session_id, "assistant", reply_text)

        elapsed = time.perf_counter() - t0
        CHAT_REQUESTS.labels(intent=intent_label or "unknown").inc()
        CHAT_LATENCY.observe(elapsed)
        if rag_ctx:
            RAG_HITS.inc()

        log.info({"event": "chat_complete", "intent": intent_label, "chars": len(reply_text), "elapsed_s": round(elapsed, 2)})
        return {"response": reply_text, "intent": intent_label, "session_id": body.session_id}

    except HTTPException:
        raise
    except Exception:
        log.exception({"event": "chat_failed"})
        raise HTTPException(status_code=503, detail="Agent invocation failed — see server logs.") from None


@app.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(request: Request, body: ChatPayload):  # noqa: ARG001
    """Server-Sent Events streaming endpoint.

    Stage 1: Run the graph with ``streaming_mode=True``. The response_node
    skips its LLM call and instead stores the assembled prompt context in
    ``state["streaming_context"]``.

    Stage 2: Stream the LLM response token-by-token using ``astream``.

    Emits ``data: {"token": "..."}`` lines, terminated by ``data: [DONE]``.
    """
    log = logging.getLogger("agent.agent_app.stream")
    memory = get_runtime().memory
    llm = get_runtime().llm_chat

    history = memory.get_history(body.session_id, 5)
    state = _build_state(body, history, streaming_mode=True)

    # Stage 1 — run graph to collect context (fast: no LLM response generation)
    result = await asyncio.to_thread(GRAPH.invoke, state)
    streaming_ctx = result.get("streaming_context") or {}
    intent_label = str(result.get("intent", ""))

    # If response_node short-circuited (e.g. upstream error), stream the text directly.
    pre_built = str(result.get("final_response", "")).strip()

    async def event_gen():
        if pre_built and not streaming_ctx:
            yield f"data: {json.dumps({'token': pre_built})}\n\n"
            yield "data: [DONE]\n\n"
            return

        system_msg = streaming_ctx.get("system", "You are a helpful finance assistant.")
        human_msg = streaming_ctx.get("human", body.message)
        full_text = []

        try:
            async for chunk in llm.astream([("system", system_msg), ("human", human_msg)]):
                token = chunk.content
                if token:
                    full_text.append(token)
                    yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as exc:
            log.warning({"event": "stream_llm_error", "detail": repr(exc)})
            yield f"data: {json.dumps({'error': 'LLM stream interrupted'})}\n\n"
        finally:
            yield "data: [DONE]\n\n"
            assembled = "".join(full_text).strip()
            if assembled:
                memory.add_message(body.session_id, "user", body.message)
                memory.add_message(body.session_id, "assistant", assembled)
                CHAT_REQUESTS.labels(intent=intent_label or "unknown").inc()
                log.info({"event": "stream_complete", "intent": intent_label, "tokens": len(full_text)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/chat/history/{session_id}")
async def history_snapshot(session_id: str):
    return {"session_id": session_id, "history": get_runtime().memory.get_full_history(session_id)}


@app.delete("/chat/history/{session_id}")
async def history_delete(session_id: str):
    get_runtime().memory.clear_session(session_id)
    return {"session_id": session_id, "status": "cleared"}


@app.get("/health")
async def health_endpoint():
    ollama_ok = check_ollama_health()
    return {"status": "ok", "service": "agent", "ollama": "reachable" if ollama_ok else "unreachable"}


@app.get("/metrics")
async def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
