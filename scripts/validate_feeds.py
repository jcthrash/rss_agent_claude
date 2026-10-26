#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import feedparser


DEFAULT_TIMEOUT = 20
DEFAULT_WORKERS = 12


@dataclass
class FeedResult:
    original_url: str
    tested_url: str
    status: str
    entry_count: int
    bozo: bool
    feed_title: str
    error: str
    elapsed_sec: float


def normalize_feed_url(url: str) -> str:
    url = url.strip()
    if url.startswith("http://"):
        return "https://" + url[len("http://"):]
    return url


def candidate_urls(url: str) -> list[str]:
    url = url.strip()
    if url.startswith("http://"):
        return [normalize_feed_url(url), url]
    parsed = urlparse(url)
    if parsed.scheme == "https":
        return [url, "http://" + url[len("https://"):]]
    return [url]


def read_feed_urls(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    urls = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    # preserve order, drop dupes
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def validate_one(url: str) -> FeedResult:
    start = time.time()
    last_error = ""

    for test_url in candidate_urls(url):
        try:
            parsed = feedparser.parse(test_url)
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            continue

        bozo = bool(getattr(parsed, "bozo", 0))
        bozo_exc = getattr(parsed, "bozo_exception", None)
        if bozo_exc:
            last_error = f"{type(bozo_exc).__name__}: {bozo_exc}"

        entries = getattr(parsed, "entries", []) or []
        feed_title = ""
        feed_obj = getattr(parsed, "feed", None)
        if feed_obj:
            feed_title = getattr(feed_obj, "title", "") or ""

        elapsed = round(time.time() - start, 2)

        if entries:
            status = "ok"
            if test_url != url:
                status = "ok_via_fallback"
            elif bozo:
                status = "ok_with_parse_warning"
            return FeedResult(
                original_url=url,
                tested_url=test_url,
                status=status,
                entry_count=len(entries),
                bozo=bozo,
                feed_title=feed_title,
                error=last_error,
                elapsed_sec=elapsed,
            )

        # No entries but parsed something.
        status = "empty"
        if test_url != url:
            status = "empty_via_fallback"
        return FeedResult(
            original_url=url,
            tested_url=test_url,
            status=status,
            entry_count=0,
            bozo=bozo,
            feed_title=feed_title,
            error=last_error,
            elapsed_sec=elapsed,
        )

    return FeedResult(
        original_url=url,
        tested_url="",
        status="failed",
        entry_count=0,
        bozo=False,
        feed_title="",
        error=last_error or "Unknown error",
        elapsed_sec=round(time.time() - start, 2),
    )


def summarize(results: Iterable[FeedResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RSS feeds and write a routine health report.")
    parser.add_argument("--feeds", default="config/feeds.txt", help="Path to feed list")
    parser.add_argument("--output-json", default="data/outputs/reports/feed_validation.json", help="JSON report output")
    parser.add_argument("--output-csv", default="data/outputs/csv/feed_validation.csv", help="CSV report output")
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT, help="Network timeout in seconds")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_WORKERS, help="Parallel worker count")
    args = parser.parse_args()

    socket.setdefaulttimeout(args.timeout_sec)

    feeds_path = Path(args.feeds)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not feeds_path.exists():
        raise SystemExit(f"Feed list not found: {feeds_path}")

    urls = read_feed_urls(feeds_path)
    if not urls:
        raise SystemExit(f"No feeds found in: {feeds_path}")

    print(f"Validating {len(urls)} feeds with {args.max_workers} workers...")

    results: list[FeedResult] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(validate_one, url): url for url in urls}
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            print(f"[{result.status:<22}] {result.original_url} -> {result.tested_url or '-'} ({result.entry_count} entries, {result.elapsed_sec}s)")
            if result.error:
                print(f"  note: {result.error}")

    results.sort(key=lambda r: (r.status, r.original_url))
    summary = summarize(results)

    payload = {
        "generated_at_epoch": time.time(),
        "feed_count": len(results),
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # lightweight CSV without pandas dependency
    import csv
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "original_url",
                "tested_url",
                "status",
                "entry_count",
                "bozo",
                "feed_title",
                "error",
                "elapsed_sec",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote JSON report: {output_json}")
    print(f"Wrote CSV report:  {output_csv}")


if __name__ == "__main__":
    main()
