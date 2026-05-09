"""Lazy runtime wiring so `rag.ingest` can import embeddings without touching Redis."""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_ollama import ChatOllama

from memory.redis_memory import RedisMemory

_runtime: "RuntimeDeps | None" = None


class RuntimeDeps:
    def __init__(self) -> None:
        self.memory = RedisMemory()
        self.llm_chat = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )


def get_runtime() -> RuntimeDeps:
    global _runtime

    if _runtime is None:
        _runtime = RuntimeDeps()

    return _runtime


@lru_cache
def embedding_model():
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
