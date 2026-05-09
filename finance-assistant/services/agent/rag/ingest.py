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
from langchain_text_splitters import RecursiveCharacterTextSplitter

from deps import embedding_model
from rag.chroma_factory import get_chroma_client

_log = logging.getLogger("agent.rag.ingest")


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


def ingest_main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    collection_name = os.environ["CHROMA_COLLECTION"]

    client = get_chroma_client()
    embedder = embedding_model()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    loaded = load_documents()

    splits: list[Document] = []
    for base in loaded:
        splits.extend(splitter.split_documents([base]))

    total_chunks = len(splits)
    print(f"Ingesting {total_chunks} chunks sourced from {len(loaded)} Markdown files...")
    _log.info({"event": "ingest_documents_split", **{"chunks": total_chunks}})

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

    _log.info({"event": "ingest_finished", **{"chunks": total_chunks, "sources": len(loaded)}})
    print(f"Ingested {total_chunks} chunks across {len(loaded)} Markdown sources")


if __name__ == "__main__":
    ingest_main()
