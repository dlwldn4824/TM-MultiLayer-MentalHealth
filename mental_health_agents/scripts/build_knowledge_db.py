#!/usr/bin/env python3
"""Build ChromaDB knowledge index from markdown files."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag import KnowledgeRAG


def main() -> None:
    import sys

    from src.config import load_config
    from src.official_fetcher import get_official_sources, load_sources_yaml

    cfg = load_config()
    kb = cfg.get("knowledge_base", {})
    if kb.get("require_fetched_official", True) and kb.get("mode", "official") != "mock":
        root = Path(__file__).resolve().parent.parent / "data/knowledge"
        missing = []
        for src in get_official_sources(load_sources_yaml()):
            p = root / src.get("file", "")
            if not p.exists() or p.stat().st_size < 100:
                missing.append(src["id"])
        if missing:
            print(
                "Missing fetched official files:",
                missing,
                "\nRun: python scripts/fetch_official_knowledge.py",
                file=sys.stderr,
            )
            sys.exit(1)

    rag = KnowledgeRAG(cfg)
    n = rag.build_index(reset=True)
    mode = kb.get("mode", "official")
    print(f"Indexed {n} chunks (mode={mode}) into {rag.chroma_dir}/{rag.collection_name}")


if __name__ == "__main__":
    main()
