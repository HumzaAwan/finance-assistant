from __future__ import annotations

import logging
import os
import re
from typing import List

from deps import embedding_model
from rag.chroma_factory import get_chroma_client

_log = logging.getLogger("agent.rag.retriever")

# Module-level singleton — avoids re-opening the Chroma PersistentClient on
# every request, which was causing unnecessary disk I/O and file-lock pressure.
_retriever_instance: "RAGRetriever | None" = None


def get_retriever() -> "RAGRetriever":
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever()
    return _retriever_instance


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def _rrf(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion across multiple ranked ID lists."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)


class RAGRetriever:
    """Hybrid BM25 + dense-vector retriever with Reciprocal Rank Fusion.

    At construction time the full corpus is loaded from Chroma to build the
    BM25 index.  The dense-vector path uses the existing Chroma query API.
    Both result lists are fused with RRF before the score threshold is applied.
    """

    def __init__(self) -> None:
        self.client = get_chroma_client()
        name = os.environ["CHROMA_COLLECTION"]
        self.collection = self.client.get_collection(name)
        self.embedder = embedding_model()
        self._build_bm25_index()
        _log.info({"event": "retriever_ready", "collection": name, "corpus_size": len(self._bm25_ids)})

    # ── BM25 index ────────────────────────────────────────────────────────────

    def _build_bm25_index(self) -> None:
        """Fetch all documents from Chroma and build an in-memory BM25 index."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            _log.warning({"event": "bm25_unavailable", "hint": "pip install rank-bm25"})
            self._bm25 = None
            self._bm25_ids: list[str] = []
            self._bm25_meta: list[dict] = []
            return

        corpus = self.collection.get(include=["documents", "metadatas"])
        docs = corpus.get("documents") or []
        ids = corpus.get("ids") or []
        metas = corpus.get("metadatas") or [{}] * len(docs)

        self._bm25_ids = list(ids)
        self._bm25_meta = list(metas)
        tokenized = [_tokenize(d) for d in docs]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        _log.info({"event": "bm25_index_built", "docs": len(docs)})

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        min_score: float = 0.20,
        topic_filter: list[str] | None = None,
    ) -> List[dict]:
        """Hybrid retrieve: vector search + BM25 fused with RRF.

        Args:
            query: Natural-language query string.
            top_k: Maximum chunks to return after fusion and filtering.
            min_score: Minimum normalised vector score (0–1) to include a chunk.
            topic_filter: If set, restrict Chroma vector search to these topics.
        """
        # ── Dense vector search ───────────────────────────────────────────────
        vector = self.embedder.embed_query(query)
        n_candidates = min(top_k * 3, max(10, self.collection.count()))
        where = {"topic": {"$in": topic_filter}} if topic_filter else None

        try:
            vec_hits = self.collection.query(
                query_embeddings=[vector],
                n_results=n_candidates,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            _log.warning({"event": "vector_search_failed", "detail": repr(exc)})
            vec_hits = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        vec_ids: list[str] = vec_hits.get("ids", [[]])[0] or []
        vec_docs: list[str] = (vec_hits.get("documents") or [[]])[0] or []
        vec_metas: list[dict] = (vec_hits.get("metadatas") or [[]])[0] or []
        vec_dists: list[float] = (vec_hits.get("distances") or [[]])[0] or []

        # Build id → (doc, meta, score) lookup from vector results
        id_to_data: dict[str, tuple[str, dict, float]] = {}
        for i, doc_id in enumerate(vec_ids):
            dist = vec_dists[i] if i < len(vec_dists) else 1.0
            score = 1.0 / (1.0 + float(dist))
            meta = vec_metas[i] if i < len(vec_metas) else {}
            doc = vec_docs[i] if i < len(vec_docs) else ""
            id_to_data[doc_id] = (doc, meta, score)

        # ── BM25 keyword search ───────────────────────────────────────────────
        bm25_ids: list[str] = []
        if self._bm25 is not None and self._bm25_ids:
            q_tokens = _tokenize(query)
            scores_arr = self._bm25.get_scores(q_tokens)
            import numpy as np
            top_indices = np.argsort(scores_arr)[::-1][:n_candidates]
            for idx in top_indices:
                if scores_arr[idx] > 0:
                    cid = self._bm25_ids[idx]
                    # Apply topic filter to BM25 results too
                    if topic_filter:
                        meta = self._bm25_meta[idx] if idx < len(self._bm25_meta) else {}
                        if meta.get("topic") not in topic_filter:
                            continue
                    bm25_ids.append(cid)

        # ── RRF fusion ────────────────────────────────────────────────────────
        ranked_lists = [vec_ids]
        if bm25_ids:
            ranked_lists.append(bm25_ids)

        fused_ids = _rrf(ranked_lists)[:top_k * 2]

        # ── Assemble results ──────────────────────────────────────────────────
        # For IDs that appeared only in BM25 (not in vector hits), fetch from Chroma
        missing = [i for i in fused_ids if i not in id_to_data]
        if missing:
            try:
                fetched = self.collection.get(ids=missing, include=["documents", "metadatas"])
                for j, fid in enumerate(fetched.get("ids") or []):
                    doc = (fetched["documents"] or [])[j] if j < len(fetched.get("documents") or []) else ""
                    meta = (fetched["metadatas"] or [])[j] if j < len(fetched.get("metadatas") or []) else {}
                    id_to_data[fid] = (doc, meta, 0.15)  # low base score for BM25-only hits
            except Exception as exc:
                _log.debug({"event": "bm25_fetch_failed", "detail": repr(exc)})

        rows: List[dict] = []
        for doc_id in fused_ids:
            if doc_id not in id_to_data:
                continue
            content, meta, score = id_to_data[doc_id]
            if score < min_score:
                _log.debug({"event": "rag_chunk_below_threshold", "score": round(score, 3), "min_score": min_score})
                continue
            rows.append({
                "content": content,
                "source": meta.get("source", "unknown"),
                "topic": meta.get("topic", ""),
                "score": round(score, 4),
            })
            if len(rows) >= top_k:
                break

        _log.info({"event": "rag_query", "results": len(rows), "hybrid": bool(bm25_ids), "min_score": min_score})
        return rows
