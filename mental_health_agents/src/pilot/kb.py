"""Small official KB (3 sources) for pilot RAG."""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.knowledge_ingestion import build_knowledge_chunks
from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_PILOT_SOURCE_IDS = ("nimh_depression", "nimh_anxiety", "who_depression")


def build_pilot_chunks(
    source_ids: list[str] | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict[str, Any]]:
    ids = set(source_ids or DEFAULT_PILOT_SOURCE_IDS)
    all_chunks = build_knowledge_chunks(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        mode="official",
        include_mock=False,
    )
    return [c for c in all_chunks if c["source_id"] in ids]


class PilotRAG:
    def __init__(self, config: dict[str, Any]):
        rag_cfg = config.get("rag", {})
        kb_cfg = config.get("knowledge_base", {})
        self.chroma_dir = str(PROJECT_ROOT / rag_cfg["chroma_dir"])
        self.collection_name = rag_cfg.get("collection_name", "counsel_pilot_kb")
        self.embedding_model = rag_cfg.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.top_k = int(kb_cfg.get("top_k", 3))
        self.source_ids = list(
            config.get("pilot", {}).get("source_ids", list(DEFAULT_PILOT_SOURCE_IDS))
        )
        self.chunk_size = int(kb_cfg.get("chunk_size", 500))
        self.chunk_overlap = int(kb_cfg.get("chunk_overlap", 50))

        self._ef = SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
        self._client = chromadb.PersistentClient(path=self.chroma_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._ef,
        )

    def build_index(self, reset: bool = True) -> int:
        if reset:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._ef,
            )

        chunks = build_pilot_chunks(
            source_ids=self.source_ids,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if not chunks:
            logger.warning("No pilot KB chunks found for sources=%s", self.source_ids)
            return 0

        self._collection.add(
            ids=[c["chunk_id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[
                {
                    "source_id": c["source_id"],
                    "title": c["title"],
                    "url": c["url"],
                    "section": c["section"],
                    "chunk_id": c["chunk_id"],
                }
                for c in chunks
            ],
        )
        logger.info("Pilot KB indexed %d chunks from %s", len(chunks), self.source_ids)
        return len(chunks)

    def retrieve_structured(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        k = top_k or self.top_k
        empty: dict[str, Any] = {"query": query, "top_k": k, "retrieved_chunks": []}
        if self._collection.count() == 0:
            return empty

        n = min(k, self._collection.count())
        results = self._collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        chunks = []
        for doc, meta, dist in zip(docs, metas, dists):
            score = float(1.0 / (1.0 + max(dist, 0.0)))
            chunks.append(
                {
                    "source_id": meta.get("source_id", ""),
                    "title": meta.get("title", ""),
                    "section": meta.get("section", ""),
                    "url": meta.get("url", ""),
                    "score": score,
                    "text": doc,
                    "chunk_id": meta.get("chunk_id", ""),
                }
            )
        return {"query": query, "top_k": k, "retrieved_chunks": chunks}
