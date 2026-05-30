"""RAG with citation-aware knowledge ingestion and structured retrieval output."""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.config import load_config
from src.knowledge_ingestion import build_knowledge_chunks

logger = logging.getLogger(__name__)


def _distance_to_score(distance: float) -> float:
    """Convert Chroma distance to similarity-like score in [0, 1]."""
    return float(1.0 / (1.0 + max(distance, 0.0)))


def evidence_texts_from_retrieval(retrieval: dict[str, Any]) -> list[str]:
    """Plain-text evidence lines for LLM prompts (backward compatible)."""
    lines = []
    for i, ch in enumerate(retrieval.get("retrieved_chunks", []), 1):
        title = ch.get("title", "")
        section = ch.get("section", "")
        url = ch.get("url", "")
        score = ch.get("score", 0)
        header = f"[{i}] {title} — {section} (score={score:.2f})"
        if url:
            header += f" <{url}>"
        lines.append(f"{header}\n{ch.get('text', '')[:800]}")
    return lines


class KnowledgeRAG:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        kb = self.config.get("knowledge_base", {})
        rag_cfg = self.config.get("rag", {})

        self.knowledge_mode = kb.get("mode", "official")
        self.include_mock = bool(kb.get("include_mock", False))
        self.chroma_dir = rag_cfg["chroma_dir"]
        base_collection = rag_cfg.get("collection_name", "mental_health_knowledge")
        self.collection_name = f"{base_collection}_{self.knowledge_mode}"
        self.embedding_model = rag_cfg.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.chunk_size = kb.get("chunk_size", rag_cfg.get("chunk_size", 500))
        self.chunk_overlap = kb.get("chunk_overlap", rag_cfg.get("chunk_overlap", 50))
        self.top_k = kb.get("top_k", rag_cfg.get("top_k", 3))

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

        chunks = build_knowledge_chunks(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            mode=self.knowledge_mode,
            include_mock=self.include_mock,
        )
        if not chunks:
            logger.warning("No knowledge chunks for mode=%s", self.knowledge_mode)
            return 0

        metadatas = [
            {
                "source_id": c["source_id"],
                "title": c["title"],
                "source_type": c["source_type"],
                "url": c["url"],
                "section": c["section"],
                "chunk_id": c["chunk_id"],
                "retrieved_at": c.get("retrieved_at", "") or "",
            }
            for c in chunks
        ]
        self._collection.add(
            ids=[c["chunk_id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=metadatas,
        )
        logger.info(
            "Indexed %d chunks (mode=%s, collection=%s)",
            len(chunks),
            self.knowledge_mode,
            self.collection_name,
        )
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

        retrieved = []
        for doc, meta, dist in zip(docs, metas, dists):
            meta = meta or {}
            retrieved.append(
                {
                    "source_id": meta.get("source_id", ""),
                    "title": meta.get("title", ""),
                    "section": meta.get("section", ""),
                    "url": meta.get("url", ""),
                    "score": _distance_to_score(float(dist)),
                    "text": doc,
                    "chunk_id": meta.get("chunk_id", ""),
                    "retrieved_at": meta.get("retrieved_at", ""),
                }
            )
        return {"query": query, "top_k": k, "retrieved_chunks": retrieved}

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        """Backward-compatible plain text evidence."""
        return evidence_texts_from_retrieval(self.retrieve_structured(query, top_k))

    def retrieve_with_sources(self, query: str, top_k: int | None = None) -> list[dict[str, str]]:
        structured = self.retrieve_structured(query, top_k)
        return [
            {
                "text": c["text"],
                "source": c.get("source_id", "unknown"),
                "title": c.get("title", ""),
                "score": str(c.get("score", 0)),
            }
            for c in structured.get("retrieved_chunks", [])
        ]
