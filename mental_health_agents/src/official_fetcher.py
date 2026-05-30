"""Fetch and parse ONLY whitelisted official URLs from sources.yaml."""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import yaml
from bs4 import BeautifulSoup

from src.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

USER_AGENT = "MentalHealthResearchBot/1.0 (academic; +local RAG ingestion)"


def load_sources_yaml(path: Path | None = None) -> dict[str, Any]:
    cfg = load_config()
    kb = cfg.get("knowledge_base", {})
    path = path or Path(kb.get("sources_file", "data/knowledge/sources.yaml"))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_official_sources(registry: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    reg = registry or load_sources_yaml()
    sources = list(reg.get("official_sources", []))
    for s in sources:
        s.setdefault("source_type", s.get("type", "official_public"))
    return sources


def get_mock_sources(registry: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    reg = registry or load_sources_yaml()
    sources = list(reg.get("mock_sources", []))
    for s in sources:
        s["source_type"] = "mock_research"
        s.setdefault("url", "")
    return sources


def _fetch_policy(registry: dict[str, Any]) -> dict[str, Any]:
    return registry.get("fetch_policy", {})


def validate_official_url(url: str, registry: dict[str, Any] | None = None) -> None:
    """Raise ValueError if URL is not on the whitelist or matches deny patterns."""
    reg = registry or load_sources_yaml()
    policy = _fetch_policy(reg)
    allowed_hosts = set(policy.get("allowed_hosts", []))
    denied = [p.lower() for p in policy.get("denied_url_patterns", [])]

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in allowed_hosts:
        raise ValueError(f"Host not allowed for fetch: {host}")

    low = url.lower()
    for pat in denied:
        if pat in low:
            raise ValueError(f"URL matches denied pattern '{pat}': {url}")

    whitelist = {s["url"].rstrip("/") for s in get_official_sources(reg)}
    if url.rstrip("/") not in whitelist:
        raise ValueError(f"URL not in official_sources whitelist: {url}")


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def html_to_sections(html: str, page_title: str) -> list[tuple[str, str]]:
    """Extract (section_title, paragraph_text) from HTML main content."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", role="main")
        or soup.body
    )
    if not main:
        return [("Overview", _clean_text(soup.get_text()))]

    sections: list[tuple[str, str]] = []
    current_title = page_title or "Overview"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        body = "\n\n".join(buffer).strip()
        if body and len(body) > 40:
            sections.append((current_title, body))
        buffer = []

    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        name = el.name
        text = _clean_text(el.get_text(separator=" "))
        if not text or len(text) < 3:
            continue
        if name in ("h1", "h2", "h3", "h4"):
            flush()
            current_title = text[:120]
        else:
            buffer.append(text)

    flush()
    if not sections:
        fallback = _clean_text(main.get_text(separator="\n"))
        if fallback:
            sections = [("Overview", fallback[:8000])]
    return sections


def sections_to_markdown(
    source: dict[str, Any],
    sections: list[tuple[str, str]],
    retrieved_at: str,
) -> str:
    lines = [
        f"# {source.get('title', source['id'])}",
        "",
        f"**Source URL:** {source['url']}",
        f"**Source ID:** {source['id']}",
        f"**Retrieved at:** {retrieved_at}",
        f"**License note:** {source.get('license_note', 'Public health information')}",
        "",
        "> Auto-fetched excerpt for research RAG. Not clinical advice. Not a substitute for the full source page.",
        "",
    ]
    for title, body in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(body)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def fetch_official_source(
    source: dict[str, Any],
    registry: dict[str, Any] | None = None,
    timeout: int | None = None,
) -> tuple[str, str]:
    """
    Fetch one whitelisted URL. Returns (markdown_content, retrieved_at ISO date).
    """
    reg = registry or load_sources_yaml()
    url = source["url"]
    validate_official_url(url, reg)

    policy = _fetch_policy(reg)
    timeout = timeout or policy.get("request_timeout_seconds", 30)
    max_chars = int(policy.get("max_chars_per_page", 15000))

    logger.info("Fetching %s", url)
    resp = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
    )
    resp.raise_for_status()

    sections = html_to_sections(resp.text, source.get("title", ""))
    retrieved_at = date.today().isoformat()

    # NICE overview: if page is JS-heavy and content thin, keep minimal overview
    if source.get("fetch_mode") == "overview_only":
        total_len = sum(len(b) for _, b in sections)
        if total_len < 400:
            sections = [
                (
                    "Guideline scope (overview)",
                    "NICE guideline NG222 covers identifying, treating and managing depression "
                    "in adults. This RAG index stores only a high-level public overview, not the "
                    "licensed full recommendation text. Refer to the official URL for complete guidance.",
                )
            ]

    md = sections_to_markdown(source, sections, retrieved_at)
    if len(md) > max_chars:
        md = md[:max_chars] + "\n\n[Truncated for research index size limit.]\n"
    return md, retrieved_at


def save_official_markdown(
    source: dict[str, Any],
    content: str,
    retrieved_at: str,
    knowledge_root: Path | None = None,
) -> Path:
    cfg = load_config()
    kb = cfg.get("knowledge_base", {})
    root = knowledge_root or Path(kb.get("root", "data/knowledge"))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    rel = source.get("file", f"official/{source['id']}.md")
    out_path = root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")

    meta_path = out_path.with_suffix(".meta.yaml")
    meta_path.write_text(
        yaml.safe_dump(
            {
                "source_id": source["id"],
                "title": source.get("title"),
                "url": source["url"],
                "retrieved_at": retrieved_at,
                "source_type": source.get("source_type", source.get("type")),
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return out_path


def fetch_all_official_sources(
    registry: dict[str, Any] | None = None,
    skip_existing: bool = False,
) -> list[dict[str, Any]]:
    """Fetch every official_sources entry and write to official/*.md."""
    reg = registry or load_sources_yaml()
    results = []
    for source in get_official_sources(reg):
        out_rel = source.get("file", "")
        root = PROJECT_ROOT / "data/knowledge"
        out_path = root / out_rel
        if skip_existing and out_path.exists() and out_path.stat().st_size > 200:
            logger.info("Skip existing %s", out_path)
            results.append({"id": source["id"], "status": "skipped", "path": str(out_path)})
            continue
        try:
            md, retrieved_at = fetch_official_source(source, reg)
            path = save_official_markdown(source, md, retrieved_at)
            source["retrieved_at"] = retrieved_at
            results.append({"id": source["id"], "status": "ok", "path": str(path), "retrieved_at": retrieved_at})
        except Exception as e:
            logger.error("Failed %s: %s", source["id"], e)
            results.append({"id": source["id"], "status": "error", "error": str(e)})
    return results
