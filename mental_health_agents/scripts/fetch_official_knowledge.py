#!/usr/bin/env python3
"""
Fetch official knowledge from whitelisted URLs in data/knowledge/sources.yaml.

Only URLs under official_sources are downloaded. No DSM/ICD full text.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.official_fetcher import fetch_all_official_sources

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch whitelisted official health pages")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sources that already have a non-empty .md file",
    )
    args = parser.parse_args()

    results = fetch_all_official_sources(skip_existing=args.skip_existing)
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(json.dumps({"fetched": ok, "total": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
