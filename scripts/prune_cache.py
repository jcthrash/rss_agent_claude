#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common_paths import (
    FILTERED_JSON,
    LLM_SCORES_JSON,
    NEW_HASHES_JSON,
    NEW_IDS_JSON,
    RAW_FEEDS_JSON,
    RANKED_JSON,
    SEMANTIC_SCORES_JSON,
    ensure_project_dirs,
    load_json,
    save_json,
)

DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prune old cached articles and synchronize derived caches.")
    p.add_argument("--max-age-days", type=int, default=30, help="Keep only articles newer than this many days")
    return p.parse_args()


def parse_timestamp(article: dict[str, Any]) -> float | None:
    ts = article.get("timestamp")
    if ts is not None:
        try:
            return float(ts)
        except Exception:
            pass

    fetched_at = article.get("fetched_at")
    if isinstance(fetched_at, str) and fetched_at.strip():
        for fmt in DATE_FORMATS:
            try:
                dt = datetime.strptime(fetched_at.strip(), fmt)
                return dt.replace(tzinfo=timezone.utc).timestamp()
            except Exception:
                continue

    published = article.get("published")
    if isinstance(published, str) and published.strip():
        for fmt in DATE_FORMATS:
            try:
                dt = datetime.strptime(published.strip(), fmt)
                return dt.replace(tzinfo=timezone.utc).timestamp()
            except Exception:
                continue

    return None


def article_key(article: dict[str, Any]) -> str:
    return str(article.get("content_hash") or article.get("id") or "")


def prune_articles(path: Path, cutoff_ts: float) -> tuple[list[dict[str, Any]], int]:
    articles = load_json(path, [])
    if not isinstance(articles, list):
        return [], 0

    kept: list[dict[str, Any]] = []
    removed = 0

    for article in articles:
        if not isinstance(article, dict):
            removed += 1
            continue

        ts = parse_timestamp(article)
        if ts is None or ts >= cutoff_ts:
            kept.append(article)
        else:
            removed += 1

    save_json(path, kept)
    return kept, removed


def filter_score_cache(path: Path, valid_keys: set[str]) -> int:
    cache = load_json(path, {})
    if not isinstance(cache, dict):
        save_json(path, {})
        return 0

    new_cache = {k: v for k, v in cache.items() if k in valid_keys}
    removed = len(cache) - len(new_cache)
    save_json(path, new_cache)
    return removed


def main() -> None:
    ensure_project_dirs()
    args = parse_args()

    cutoff_ts = time.time() - (args.max_age_days * 86400)

    raw_articles, raw_removed = prune_articles(RAW_FEEDS_JSON, cutoff_ts)
    filtered_articles, filtered_removed = prune_articles(FILTERED_JSON, cutoff_ts)
    ranked_articles, ranked_removed = prune_articles(RANKED_JSON, cutoff_ts)

    valid_raw_ids = {str(a.get("id")) for a in raw_articles if a.get("id")}
    valid_raw_hashes = {str(a.get("content_hash")) for a in raw_articles if a.get("content_hash")}
    valid_filtered_keys = {article_key(a) for a in filtered_articles if article_key(a)}
    valid_ranked_keys = {article_key(a) for a in ranked_articles if article_key(a)}

    semantic_removed = filter_score_cache(SEMANTIC_SCORES_JSON, valid_filtered_keys | valid_raw_hashes | valid_raw_ids)
    llm_removed = filter_score_cache(LLM_SCORES_JSON, valid_ranked_keys)

    new_ids = load_json(NEW_IDS_JSON, [])
    if isinstance(new_ids, list):
        new_ids = [x for x in new_ids if str(x) in valid_raw_ids]
        save_json(NEW_IDS_JSON, new_ids)
    else:
        save_json(NEW_IDS_JSON, [])

    new_hashes = load_json(NEW_HASHES_JSON, [])
    if isinstance(new_hashes, list):
        new_hashes = [x for x in new_hashes if str(x) in valid_raw_hashes]
        save_json(NEW_HASHES_JSON, new_hashes)
    else:
        save_json(NEW_HASHES_JSON, [])

    print(f"Pruned raw cache:      removed {raw_removed}, kept {len(raw_articles)}")
    print(f"Pruned filtered cache: removed {filtered_removed}, kept {len(filtered_articles)}")
    print(f"Pruned ranked cache:   removed {ranked_removed}, kept {len(ranked_articles)}")
    print(f"Pruned semantic cache: removed {semantic_removed}")
    print(f"Pruned LLM cache:      removed {llm_removed}")


if __name__ == "__main__":
    main()
