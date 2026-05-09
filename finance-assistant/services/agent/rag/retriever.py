from __future__ import annotations

import logging
import os
from typing import List

from deps import embedding_model
from rag.chroma_factory import get_chroma_client

_log = logging.getLogger("agent.rag.retriever")


class RAGRetriever:
    def __init__(self) -> None:
        self.client = get_chroma_client()
        name = os.environ["CHROMA_COLLECTION"]
        self.collection = self.client.get_collection(name)
        self.embedder = embedding_model()
        _log.info({"event": "retriever_ready", "collection": name})

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        vector = self.embedder.embed_query(query)

        hits = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        rows: List[dict] = []
        if not hits.get("documents"):
            return rows

        chunk_container = hits["documents"][0] if hits["documents"] else []
        distances = hits.get("distances", [None])[0] if hits.get("distances") else []
        meta_container = hits.get("metadatas", [None])[0] if hits.get("metadatas") else []

        for idx, content in enumerate(chunk_container):
            dist = distances[idx] if distances and idx < len(distances) else None

            if dist is None:
                score = 0.0
            else:
                score = 1.0 / (1.0 + float(dist))
            meta = meta_container[idx] if meta_container and idx < len(meta_container) else {}

            rows.append({"content": content, "source": meta.get("source", "unknown"), "score": score})

        _log.info({"event": "rag_query", **{"results": len(rows)}})
        return rows
