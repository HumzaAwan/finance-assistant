"""Build a Chroma client: embedded disk (persist) or remote HTTP."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import chromadb

_log = logging.getLogger("agent.rag.chroma_factory")


def get_chroma_client():
    """CHROMA_MODE=persist|http — persist uses CHROMA_PERSIST_DIR or local_data/chroma."""

    mode = os.getenv("CHROMA_MODE", "persist").strip().lower()
    host_raw = os.getenv("CHROMA_HOST", "").strip()

    # Old Docker Compose .env used CHROMA_HOST=chromadb; that hostname doesn't resolve on the host.
    if mode == "http" and host_raw.lower() in {"chromadb", "chroma"}:
        _log.warning(
            {"event": "chroma_mode_coerce", "reason": "docker_hostname_on_host", **{"host": host_raw}}
        )
        mode = "persist"

    if mode in ("persist", "persistent", "local"):
        raw = os.getenv("CHROMA_PERSIST_DIR", "").strip()
        if raw:
            path = Path(raw)
        else:
            path = Path(__file__).resolve().parents[3] / "local_data" / "chroma"
        path.mkdir(parents=True, exist_ok=True)
        _log.info({"event": "chroma_client", "mode": "persist", "path": str(path)})
        return chromadb.PersistentClient(path=str(path))

    host = os.environ["CHROMA_HOST"]
    port = int(os.environ["CHROMA_PORT"])
    return chromadb.HttpClient(host=host, port=port)
