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

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def get_retriever() -> "RAGRetriever":
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever()
    return _retriever_instance


def reset_retriever() -> None:
    """Force re-initialisation on next call (used after ingest)."""
    global _retriever_instance
    _retriever_instance = None


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


def _reranker_enabled() -> bool:
    """Return True if RERANKER_ENABLED is not explicitly 'false' and sentence-transformers is available."""
    env_flag = os.getenv("RERANKER_ENABLED", "true").lower().strip()
    if env_flag == "false":
        return False
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


class RAGRetriever:
    """Hybrid BM25 + dense-vector retriever with Reciprocal Rank Fusion and
    optional cross-encoder reranking.

    At construction time the full corpus is loaded from Chroma to build the
    BM25 index.  The dense-vector path uses the existing Chroma query API.
    Both result lists are fused with RRF before the optional cross-encoder
    reranker rescores and re-orders candidates.
    """

    def __init__(self) -> None:
        self.client = get_chroma_client()
        name = os.environ["CHROMA_COLLECTION"]
        self.collection = self.client.get_collection(name)
        self.embedder = embedding_model()
        self._build_bm25_index()
        self._reranker = None
        self._reranker_enabled = _reranker_enabled()
        if self._reranker_enabled:
            self._load_reranker()
        _log.info({
            "event": "retriever_ready",
            "collection": name,
            "corpus_size": len(self._bm25_ids),
            "reranker_enabled": self._reranker_enabled,
            "reranker_model": RERANKER_MODEL if self._reranker_enabled else None,
        })

    # ── Reranker ──────────────────────────────────────────────────────────────

    def _load_reranker(self) -> None:
        """Load cross-encoder reranker once at init; disable gracefully on failure."""
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
            _log.info({"event": "reranker_loaded", "model": RERANKER_MODEL})
        except Exception as exc:
            _log.warning({"event": "reranker_load_failed", "detail": repr(exc), "hint": "pip install sentence-transformers"})
            self._reranker = None
            self._reranker_enabled = False

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Score (query, chunk_text) pairs with the cross-encoder; sort descending."""
        if self._reranker is None or not candidates:
            return candidates

        pairs = [(query, c["content"]) for c in candidates]
        try:
            scores: list[float] = self._reranker.predict(pairs).tolist()
        except Exception as exc:
            _log.warning({"event": "reranker_predict_failed", "detail": repr(exc)})
            return candidates

        for chunk, score in zip(candidates, scores):
            chunk["reranker_score"] = round(float(score), 4)

        reranked = sorted(candidates, key=lambda c: c.get("reranker_score", 0.0), reverse=True)

        _log.info({
            "event": "reranker_scored",
            "n": len(reranked),
            "top_score": reranked[0].get("reranker_score") if reranked else None,
            "top_source": reranked[0].get("source") if reranked else None,
        })
        return reranked

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
        """Hybrid retrieve: vector search + BM25 fused with RRF + optional reranker.

        Args:
            query: Natural-language query string.
            top_k: Maximum chunks to return after fusion, reranking, and filtering.
            min_score: Minimum normalised vector score (0–1) to include a chunk
                       in the pre-reranker candidate set.
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

        # Widen candidate pool for reranker (it needs more to choose from)
        pool_size = top_k * 4 if self._reranker_enabled else top_k * 2
        fused_ids = _rrf(ranked_lists)[:pool_size]

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

        # ── Cross-encoder reranking ───────────────────────────────────────────
        if self._reranker_enabled and rows:
            rows = self._rerank(query, rows)

        # Apply final top-K after reranking
        rows = rows[:top_k]

        _log.info({
            "event": "rag_query",
            "results": len(rows),
            "hybrid": bool(bm25_ids),
            "reranked": self._reranker_enabled,
            "min_score": min_score,
        })
        return rows

    @property
    def health_info(self) -> dict:
        """Summary for /health endpoint."""
        return {
            "reranker_enabled": self._reranker_enabled,
            "reranker_model": RERANKER_MODEL if self._reranker_enabled else None,
            "corpus_size": len(self._bm25_ids),
        }
