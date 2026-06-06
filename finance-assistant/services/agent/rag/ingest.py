from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from local_env import apply_local_dev_url_overrides

apply_local_dev_url_overrides()

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from deps import embedding_model
from rag.chroma_factory import get_chroma_client
from rag.retriever import _retriever_instance  # noqa: F401 — reset singleton after reingest

_log = logging.getLogger("agent.rag.ingest")

# Headers used by MarkdownHeaderTextSplitter to honour document structure.
# These map markdown heading levels to metadata keys in the resulting chunks.
_MD_HEADERS = [("##", "section"), ("###", "subsection")]

# 800-char chunks with 20% overlap (160 chars) ensures that named rules
# (e.g. "50/30/20") and their explanations land in the same chunk.
_CHAR_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=160,
    separators=["\n\n", "\n", ". ", " "],
)
_MD_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=_MD_HEADERS,
    strip_headers=False,
)


def documents_dir() -> Path:
    return Path(__file__).resolve().parent / "documents"


def topic_from_filename(name: str) -> str:
    stem = Path(name).stem.lower()
    return stem.replace("-", "_").replace(" ", "_")


def load_documents() -> list[Document]:
    docs: list[Document] = []
    for path in sorted(documents_dir().glob("*.md")):
        text = path.read_text(encoding="utf-8")
        meta = {"source": path.name, "topic": topic_from_filename(path.name)}
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def split_document(doc: Document) -> list[Document]:
    """Two-pass splitting: respect markdown headers first, then character limits.

    The header pass captures section context into chunk metadata so
    topic-filtered Chroma queries can match on ``section`` field.
    """
    base_meta = dict(doc.metadata)

    # Pass 1 — structural split on headers
    header_splits = _MD_SPLITTER.split_text(doc.page_content)

    # Pass 2 — enforce max character size, merging base + header metadata
    result: list[Document] = []
    for hs in header_splits:
        merged_meta = {**base_meta, **hs.metadata}
        char_splits = _CHAR_SPLITTER.split_documents(
            [Document(page_content=hs.page_content, metadata=merged_meta)]
        )
        result.extend(char_splits)

    # Fallback: if header parsing produced nothing, use char splitter directly
    if not result:
        result = _CHAR_SPLITTER.split_documents([doc])

    return result


def ingest_main() -> None:
    import rag.retriever as _ret_mod

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    collection_name = os.environ["CHROMA_COLLECTION"]

    client = get_chroma_client()
    embedder = embedding_model()

    loaded = load_documents()

    splits: list[Document] = []
    for base in loaded:
        splits.extend(split_document(base))

    total_chunks = len(splits)
    print(f"Ingesting {total_chunks} chunks sourced from {len(loaded)} Markdown files...")
    _log.info({"event": "ingest_documents_split", "chunks": total_chunks, "sources": len(loaded)})

    try:
        client.delete_collection(name=collection_name)
        _log.info({"event": "collection_deleted", "collection": collection_name})
    except Exception as exc:
        _log.warning({"event": "collection_delete_skipped", "detail": repr(exc)})

    collection = client.create_collection(name=collection_name)

    BATCH = 48
    for start in range(0, len(splits), BATCH):
        batch = splits[start : start + BATCH]

        embeddings = embedder.embed_documents([doc.page_content for doc in batch])
        texts = [doc.page_content for doc in batch]

        ids = []
        metadata: list[Any] = []
        for idx, doc in enumerate(batch):
            fname = Path(doc.metadata.get("source", "unknown")).stem
            doc_id = f"{fname}-{start + idx}"
            ids.append(doc_id)
            metadata.append(dict(doc.metadata))

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadata,
        )

    _log.info({"event": "ingest_finished", "chunks": total_chunks, "sources": len(loaded)})
    print(f"Ingested {total_chunks} chunks across {len(loaded)} Markdown sources")

    # Reset the RAGRetriever singleton so the next query picks up the new index.
    _ret_mod._retriever_instance = None
    _log.info({"event": "retriever_singleton_reset"})


if __name__ == "__main__":
    ingest_main()
