from __future__ import annotations

import json
import logging
from typing import List

from rag.retriever import get_retriever

_log = logging.getLogger("agent.tools.rag_tools")


def curated_snippets(query: str, top_k: int = 3) -> List[dict]:
    payload = get_retriever().retrieve(query=query, top_k=top_k)
    _log.info("%s", json.dumps({"event": "rag_tool_query", "hits": len(payload)}))
    return payload
