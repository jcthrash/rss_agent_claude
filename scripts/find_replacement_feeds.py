#!/usr/bin/env python3
"""
find_replacement_feeds.py

One-off tool to find working RSS replacement URLs for dead/empty feeds.
Uses Claude with web search to locate current feed URLs, then validates
each candidate before accepting it.

Usage:
    python scripts/find_replacement_feeds.py
    python scripts/find_replacement_feeds.py --validation-report data/outputs/reports/feed_validation.json
    python scripts/find_replacement_feeds.py --dry-run   # show replacements without editing feeds.txt
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import anthropic
import feedparser
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_VALIDATION_REPORT = PROJECT_ROOT / "data/outputs/reports/feed_validation.json"
DEFAULT_FEEDS_TXT = PROJECT_ROOT / "config/feeds.txt"
DEFAULT_REPORT_OUT = PROJECT_ROOT / "data/outputs/reports/feed_replacements.json"

DEAD_STATUSES = {"empty", "empty_via_fallback", "failed"}
CLAUDE_MODEL = "claude-sonnet-4-6"  # Sonnet for web search; Haiku doesn't support it
DEFAULT_DELAY = 15.0   # seconds between calls — web search is token-heavy
MAX_RETRIES = 4        # retry attempts on rate limit

client = anthropic.Anthropic()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find replacement RSS URLs for dead feeds using Claude + web search")
    p.add_argument("--validation-report", type=Path, default=DEFAULT_VALIDATION_REPORT)
    p.add_argument("--feeds", type=Path, default=DEFAULT_FEEDS_TXT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--dry-run", action="store_true", help="Print replacements but don't edit feeds.txt")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Seconds to wait between Claude calls (default: 15)")
    p.add_argument("--statuses", nargs="+", default=list(DEAD_STATUSES),
                   help="Feed statuses to treat as dead (default: empty empty_via_fallback failed)")
    return p.parse_args()


# ── Feed validation report ─────────────────────────────────────────────────────

def load_dead_feeds(report_path: Path, dead_statuses: set[str]) -> list[dict]:
    if not report_path.exists():
        raise FileNotFoundError(
            f"Validation report not found: {report_path}\n"
            "Run: python scripts/validate_feeds.py --feeds config/feeds.txt"
        )
    data = json.loads(report_path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    return [r for r in results if r.get("status") in dead_statuses]


# ── RSS validation ─────────────────────────────────────────────────────────────

def validate_feed_url(url: str, timeout: int = 20) -> tuple[bool, int]:
    """Returns (is_valid, entry_count). Tries the URL directly with feedparser."""
    try:
        parsed = feedparser.parse(url)
        entries = list(getattr(parsed, "entries", []) or [])
        bozo = bool(getattr(parsed, "bozo", False))
        if entries and not bozo:
            return True, len(entries)
        if entries:
            # bozo but has entries — still usable
            return True, len(entries)
        return False, 0
    except Exception:
        return False, 0


# ── Journal name extraction ────────────────────────────────────────────────────

def journal_name_from_url(url: str) -> str:
    """
    Best-effort extraction of a human-readable journal/publisher name from a feed URL.
    e.g. https://microbiomejournal.biomedcentral.com/... -> "microbiomejournal (biomedcentral)"
    """
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower().replace("www.", "")
        # Take the leftmost subdomain if present (most specific), plus the domain
        parts = host.split(".")
        if len(parts) >= 3:
            return f"{parts[0]} ({'.'.join(parts[1:])})"
        # Fall back to path segments for publisher-style URLs like
        # royalsocietypublishing.org/journal/rsob
        path_parts = [p for p in parsed.path.strip("/").split("/") if p and len(p) > 2]
        if path_parts:
            return f"{host} / {path_parts[0]}"
        return host
    except Exception:
        return url


# ── Claude web search ──────────────────────────────────────────────────────────

def find_replacement_url(dead_feed: dict, delay: float) -> dict:
    """
    Ask Claude (with web search) to find the current RSS URL for a dead feed.
    Returns a result dict with keys: original_url, suggested_url, reasoning, validated, entry_count.
    Retries with exponential backoff on rate limit errors.
    """
    original_url = dead_feed["original_url"]
    feed_title = dead_feed.get("feed_title", "").strip()
    status = dead_feed.get("status", "")

    # Build journal hint from title if available, otherwise derive from URL
    if feed_title:
        journal_hint = f'The feed previously had the title: "{feed_title}".'
    else:
        derived = journal_name_from_url(original_url)
        journal_hint = f'The journal/publisher can be inferred from the URL as: "{derived}".'

    prompt = f"""I have a dead or empty RSS feed that I need to replace.

Dead feed URL: {original_url}
Status: {status}
{journal_hint}

Please search the web to find the current, working RSS feed URL for this journal or publication.

Rules:
- Return ONLY the raw RSS/Atom feed URL (ending in .xml, .rss, .atom, or similar), not a webpage URL
- The URL must be a direct feed link that feedparser can parse
- If you find multiple candidate URLs, pick the most likely correct one
- If you genuinely cannot find a replacement, say exactly: NO_REPLACEMENT_FOUND

Respond with just the URL or NO_REPLACEMENT_FOUND, nothing else."""

    messages = [{"role": "user", "content": prompt}]
    tools = [{"type": "web_search_20260209", "name": "web_search"}]
    text = ""

    for attempt in range(MAX_RETRIES):
        try:
            # Agentic loop: Claude may search multiple times before answering
            max_iterations = 8
            for _ in range(max_iterations):
                response = client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=512,
                    tools=tools,
                    messages=messages,
                )
                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "end_turn":
                    for block in response.content:
                        if hasattr(block, "text"):
                            text = block.text.strip()
                    break

                if response.stop_reason == "tool_use":
                    # Web search executed server-side; results already in response.content
                    continue

                break  # unexpected stop reason
            break  # success — exit retry loop

        except anthropic.RateLimitError:
            if attempt == MAX_RETRIES - 1:
                print(f"\n  Rate limit hit {MAX_RETRIES} times; skipping this feed.")
                text = ""
                break
            backoff = delay * (2 ** attempt)
            print(f"\n  Rate limit hit; waiting {backoff:.0f}s before retry {attempt + 2}/{MAX_RETRIES}...")
            time.sleep(backoff)
            # Reset messages for a fresh attempt
            messages = [{"role": "user", "content": prompt}]

    # Parse the answer
    suggested_url = None
    if text and text != "NO_REPLACEMENT_FOUND" and (
        text.startswith("http://") or text.startswith("https://")
    ):
        suggested_url = text.rstrip(".,;)")

    return {
        "original_url": original_url,
        "feed_title": feed_title,
        "status": status,
        "suggested_url": suggested_url,
        "reasoning": text,
        "validated": False,
        "entry_count": 0,
    }


# ── feeds.txt editing ──────────────────────────────────────────────────────────

def read_feeds_txt(path: Path) -> list[str]:
    """Read feeds.txt preserving comments and blank lines."""
    return path.read_text(encoding="utf-8").splitlines()


def write_feeds_txt(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_replacements(
    feeds_path: Path,
    replacements: list[dict],
    dry_run: bool,
) -> int:
    """
    Replace dead URLs in feeds.txt with validated replacements.
    Returns count of replacements made.
    """
    lines = read_feeds_txt(feeds_path)
    url_to_replacement = {
        r["original_url"]: r["suggested_url"]
        for r in replacements
        if r.get("validated") and r.get("suggested_url")
    }

    if not url_to_replacement:
        print("No validated replacements to apply.")
        return 0

    new_lines = []
    replaced = 0
    for line in lines:
        stripped = line.strip()
        if stripped in url_to_replacement:
            new_url = url_to_replacement[stripped]
            print(f"  REPLACE: {stripped}")
            print(f"      WITH: {new_url}")
            new_lines.append(new_url)
            replaced += 1
        else:
            new_lines.append(line)

    if not dry_run:
        # Back up the original
        backup_path = feeds_path.with_suffix(".txt.bak")
        backup_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nBacked up original feeds.txt to {backup_path}")
        write_feeds_txt(feeds_path, new_lines)
        print(f"Updated {feeds_path} with {replaced} replacement(s).")
    else:
        print(f"\n[dry-run] Would have replaced {replaced} URL(s) in {feeds_path}.")

    return replaced


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    dead_statuses = set(args.statuses)

    print(f"Loading validation report from {args.validation_report}...")
    dead_feeds = load_dead_feeds(args.validation_report, dead_statuses)

    if not dead_feeds:
        print(f"No feeds with status in {dead_statuses} found. Nothing to do.")
        return

    print(f"Found {len(dead_feeds)} dead/empty feed(s) to investigate.\n")

    results = []
    for i, feed in enumerate(dead_feeds, start=1):
        url = feed["original_url"]
        title = feed.get("feed_title") or "(no title)"
        print(f"[{i}/{len(dead_feeds)}] {url}")
        print(f"  Title : {title}  |  Status: {feed.get('status')}")

        result = find_replacement_url(feed, args.delay)

        if result["suggested_url"]:
            print(f"  Candidate URL: {result['suggested_url']}")
            print("  Validating...", end=" ", flush=True)
            is_valid, entry_count = validate_feed_url(result["suggested_url"])
            result["validated"] = is_valid
            result["entry_count"] = entry_count
            if is_valid:
                print(f"✓  ({entry_count} entries)")
            else:
                print("✗  (no entries returned)")
        else:
            print("  → No replacement found.")

        results.append(result)

        if i < len(dead_feeds):
            time.sleep(args.delay)

    # Summary
    found = [r for r in results if r.get("suggested_url")]
    validated = [r for r in results if r.get("validated")]
    not_found = [r for r in results if not r.get("suggested_url")]

    print(f"\n{'='*60}")
    print(f"Results: {len(validated)} validated replacements, "
          f"{len(found) - len(validated)} candidates failed validation, "
          f"{len(not_found)} not found.")

    if not_found:
        print("\nNo replacement found for:")
        for r in not_found:
            print(f"  {r['original_url']}  ({r.get('feed_title') or 'no title'})")

    # Save report
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull report saved to {args.report_out}")

    # Apply replacements
    if validated:
        print(f"\nApplying {len(validated)} validated replacement(s) to {args.feeds}...")
        if args.dry_run:
            print("[dry-run mode — no files will be changed]\n")
        apply_replacements(args.feeds, results, dry_run=args.dry_run)
    else:
        print("\nNo validated replacements to apply to feeds.txt.")


if __name__ == "__main__":
    main()
