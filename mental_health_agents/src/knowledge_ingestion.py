"""Load sources.yaml and build chunked knowledge with citation metadata."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from src.config import PROJECT_ROOT, load_config
from src.official_fetcher import get_mock_sources, get_official_sources, load_sources_yaml


def load_sources_registry(sources_path: Path | None = None) -> list[dict[str, Any]]:
    """Merged mock + official sources for indexing."""
    reg = load_sources_yaml(sources_path)
    mock = get_mock_sources(reg)
    official = get_official_sources(reg)
    for s in official:
        meta_path = _meta_path_for_source(s)
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = yaml.safe_load(f) or {}
            s["retrieved_at"] = meta.get("retrieved_at", "")
    return mock + official


def _meta_path_for_source(source: dict[str, Any]) -> Path:
    cfg = load_config()
    root = Path(cfg.get("knowledge_base", {}).get("root", "data/knowledge"))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    rel = source.get("file", f"official/{source['id']}.md")
    return (root / rel).with_suffix(".meta.yaml")


def _parse_sections(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"(?m)^##\s+", text)
    if len(parts) <= 1:
        return [("Overview", text.strip())]
    sections = []
    first = parts[0].strip()
    if first:
        title_line = first.split("\n")[0].replace("#", "").strip()
        sections.append((title_line or "Introduction", first))
    for part in parts[1:]:
        lines = part.strip().split("\n", 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        if body:
            sections.append((title, body))
    return sections


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = end - overlap
    return chunks


def select_sources_for_mode(
    sources: list[dict[str, Any]],
    mode: str,
    include_mock: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for src in sources:
        stype = src.get("source_type", src.get("type", ""))
        is_mock = stype == "mock_research" or str(src.get("file", "")).startswith("mock/")
        if mode == "mock":
            if is_mock:
                selected.append(src)
        elif mode == "official":
            if not is_mock:
                selected.append(src)
        elif mode == "both":
            if is_mock and not include_mock:
                continue
            selected.append(src)
        else:
            if is_mock and not include_mock:
                continue
            selected.append(src)
    return selected


def build_knowledge_chunks(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    mode: str | None = None,
    include_mock: bool | None = None,
) -> list[dict[str, Any]]:
    cfg = load_config()
    kb = cfg.get("knowledge_base", {})
    rag = cfg.get("rag", {})
    chunk_size = chunk_size or kb.get("chunk_size", rag.get("chunk_size", 500))
    chunk_overlap = chunk_overlap or kb.get("chunk_overlap", rag.get("chunk_overlap", 50))
    mode = mode or kb.get("mode", "official")
    include_mock = include_mock if include_mock is not None else kb.get("include_mock", False)

    root = Path(kb.get("root", "data/knowledge"))
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    sources = select_sources_for_mode(load_sources_registry(), mode, include_mock)
    all_chunks: list[dict[str, Any]] = []

    for src in sources:
        rel = src.get("file", "")
        if not rel:
            continue
        md_path = root / rel
        if not md_path.exists():
            continue
        text = md_path.read_text(encoding="utf-8")
        source_id = src["id"]
        retrieved_at = src.get("retrieved_at", "")
        for sec_idx, (section_title, section_body) in enumerate(_parse_sections(text)):
            for i, chunk in enumerate(chunk_text(section_body, chunk_size, chunk_overlap)):
                chunk_id = f"{source_id}__s{sec_idx}__c{i}"
                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": chunk,
                        "source_id": source_id,
                        "title": src.get("title", source_id),
                        "source_type": src.get("source_type", src.get("type", "unknown")),
                        "url": src.get("url", "") or "",
                        "section": section_title,
                        "retrieved_at": retrieved_at,
                    }
                )
    return all_chunks
