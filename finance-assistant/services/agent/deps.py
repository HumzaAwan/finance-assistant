"""Lazy runtime wiring so `rag.ingest` can import embeddings without touching Redis."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import httpx
from langchain_ollama import ChatOllama

from memory.redis_memory import RedisMemory

_log = logging.getLogger("agent.deps")
_runtime: "RuntimeDeps | None" = None


class RuntimeDeps:
    def __init__(self) -> None:
        self.memory = RedisMemory()
        self.llm_chat = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
            timeout=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
        )


def get_runtime() -> RuntimeDeps:
    global _runtime

    if _runtime is None:
        _runtime = RuntimeDeps()

    return _runtime


def check_ollama_health() -> bool:
    """Ping Ollama at startup. Returns True if reachable, False otherwise.

    Logs a warning (not an error) on failure so the service still starts and
    serves requests — the per-request retry in llm_utils will handle transient
    outages gracefully.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        r.raise_for_status()
        models = [m.get("name", "") for m in r.json().get("models", [])]
        _log.info({"event": "ollama_ready", "url": base_url, "models": models})
        return True
    except Exception as exc:
        _log.warning(
            {
                "event": "ollama_unreachable",
                "url": base_url,
                "detail": repr(exc),
                "hint": "Start Ollama and run: ollama pull llama3.2 && ollama pull nomic-embed-text",
            }
        )
        return False


@lru_cache
def embedding_model():
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
