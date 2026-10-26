#!/usr/bin/env python3
"""
Fetch RSS/Atom feeds, supplement with Crossref API results, and pull
high-volume sources (bioRxiv via Crossref DOI prefix, Nature journals
via Crossref ISSN, ASM Press via Crossref DOI prefix) — all in one pass.

Replaces the former supplement_high_volume.py (now merged here).
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import html
import json
import re
import socket
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import feedparser

from common_paths import (
    FEEDS_TXT,
    FETCH_STATE_JSON,
    INTEREST_PROFILE_TXT,
    CONFIG_DIR,
    NEW_HASHES_JSON,
    NEW_IDS_JSON,
    NEW_PUBLISHED_IDS_JSON,
    RAW_FEEDS_JSON,
    SEEN_IDS_TXT,
    ensure_project_dirs,
    load_json,
    save_json,
)

# ── RSS constants ──────────────────────────────────────────────────────────────
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
READ_MORE_RE = re.compile(
    r"\b(read more|full article|continue reading|view full text|click here|open access)\b", re.I
)
BOILERPLATE_RE = re.compile(
    r"\b(share this article|copyright|rights reserved|supplementary material)\b", re.I
)
MAX_CONTENT_CHARS = 6000
DEFAULT_TIMEOUT_SEC = 45
DEFAULT_MAX_WORKERS = 4
DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)

# ── Crossref constants ─────────────────────────────────────────────────────────
CROSSREF_API_URL = "https://api.crossref.org/works"
DEFAULT_CROSSREF_DAYS_BACK = 21
DEFAULT_CROSSREF_ROWS = 100
DEFAULT_CROSSREF_MAX_PER_TERM = 150
CROSSREF_USER_AGENT = "RSS-Agent/1.0 (mailto:{mailto})"

# ── High-volume source constants (formerly supplement_high_volume.py) ──────────
BIORXIV_PREFIXES = ["10.1101", "10.64898"]
ASM_PREFIXES = ["10.1128"]       # ASM Press — RSS blocked by Cloudflare; fetch via DOI prefix
CURSOR_PAGE_SIZE = 1000
CURSOR_REQUEST_DELAY = 0.5
DEFAULT_BIORXIV_TERMS = CONFIG_DIR / "biorxiv_terms.txt"
DEFAULT_NATURE_JOURNALS = CONFIG_DIR / "nature_journals.txt"

socket.setdefaulttimeout(DEFAULT_TIMEOUT_SEC)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch RSS feeds and supplement from Crossref")
    # shared
    p.add_argument("--feeds", type=Path, default=FEEDS_TXT)
    p.add_argument("--raw-cache", type=Path, default=RAW_FEEDS_JSON)
    p.add_argument("--new-ids", type=Path, default=NEW_IDS_JSON)
    p.add_argument("--new-published-ids", type=Path, default=NEW_PUBLISHED_IDS_JSON)
    p.add_argument("--new-hashes", type=Path, default=NEW_HASHES_JSON)
    p.add_argument("--fetch-state", type=Path, default=FETCH_STATE_JSON)
    p.add_argument("--last-run-utc", default="")
    p.add_argument("--run-started-utc", default="")
    # RSS-specific
    p.add_argument("--seen-ids", type=Path, default=SEEN_IDS_TXT)
    p.add_argument("--max-content-chars", type=int, default=MAX_CONTENT_CHARS)
    p.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p.add_argument("--force-refresh", action="store_true")
    # Crossref-specific
    p.add_argument("--interest-profile", type=Path, default=INTEREST_PROFILE_TXT)
    p.add_argument("--crossref-terms", type=Path, default=None,
                   help="Path to Crossref query terms file (default: uses --interest-profile)")
    p.add_argument("--crossref-days-back", type=int, default=DEFAULT_CROSSREF_DAYS_BACK)
    p.add_argument("--crossref-rows", type=int, default=DEFAULT_CROSSREF_ROWS)
    p.add_argument("--crossref-max-per-term", type=int, default=DEFAULT_CROSSREF_MAX_PER_TERM)
    p.add_argument("--crossref-mailto", default="")
    p.add_argument("--skip-crossref", action="store_true", help="Skip Crossref supplementation")
    # High-volume sources (bioRxiv, Nature)
    p.add_argument("--biorxiv-terms", type=Path, default=DEFAULT_BIORXIV_TERMS)
    p.add_argument("--nature-journals", type=Path, default=DEFAULT_NATURE_JOURNALS)
    p.add_argument("--skip-biorxiv", action="store_true")
    p.add_argument("--skip-nature", action="store_true")
    p.add_argument("--skip-asm", action="store_true", help="Skip ASM DOI prefix fetch")
    return p.parse_args()


# ── Shared utilities ───────────────────────────────────────────────────────────

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc_string(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(value, fmt)
            dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
            return dt.timestamp()
        except Exception:
            continue
    try:
        dt = parsedate_to_datetime(value)
        dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def iso_from_ts(ts: float | None) -> str:
    return "" if ts is None else datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_text(s: str, n: int = 24) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()[:n]


def normalize_title(title: str) -> str:
    return WS_RE.sub(" ", (title or "").strip().lower())


def normalize_link(link: str) -> str:
    link = (link or "").strip()
    if not link:
        return ""
    parsed = urlparse(link)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")


# ── DOI extraction for cross-source dedup ──────────────────────────────────────
# The same paper fetched via Crossref (doi.org/10.1128/...) and RSS
# (journals.asm.org/doi/abs/10.1128/...) gets different content_hashes and
# normalized_links.  Extracting the DOI provides a stable identifier that
# matches across sources.
_DOI_RE = re.compile(r'\b(10\.\d{4,9}/[^\s?#<>"\'\]]+)', re.IGNORECASE)

# Publisher URL → DOI fallbacks (when the DOI isn't literally in the URL).
# Each tuple is (compiled regex matching the URL, DOI prefix to prepend to the
# captured group).  Add new publishers here as needed — no logic changes required.
_PUBLISHER_DOI_PATTERNS = [
    # nature.com/articles/<id> → 10.1038/<id>
    (re.compile(r'https?://(?:www\.)?nature\.com/articles/([a-z0-9._-]+)', re.IGNORECASE),
     '10.1038/'),
]


def extract_doi(url: str) -> str:
    """Extract a DOI from a URL, normalised for dedup.

    Returns a 'doi:' prefixed string for set membership testing, or '' if
    no DOI is found.  Strips trailing version suffixes (bioRxiv v1/v2),
    format suffixes (/full, /abstract), and OUP numeric article IDs.
    """
    if not url:
        return ""
    m = _DOI_RE.search(url)
    if not m:
        # Check publisher-specific URL patterns
        for pattern, prefix in _PUBLISHER_DOI_PATTERNS:
            pm = pattern.match(url)
            if pm:
                return 'doi:' + prefix + pm.group(1).lower()
        return ""
    doi = m.group(1).lower().rstrip('.,;:)')
    doi = re.sub(r'/(full|abstract|epdf|pdf)$', '', doi)
    doi = re.sub(r'v\d+$', '', doi)       # bioRxiv version
    doi = re.sub(r'/\d+$', '', doi)        # OUP trailing article ID
    return 'doi:' + doi


def should_mark_new_published(
    article: dict[str, Any], prev_run_ts: float | None, run_started_ts: float
) -> bool:
    # Use fetch timestamp (when we first saw it) rather than published_ts
    # (when it was originally published). RSS articles often have old
    # publication dates even when they first appear in the feed today.
    fetch_ts = article.get("timestamp")
    if fetch_ts is not None:
        try:
            fetch_ts = float(fetch_ts)
            # Don't upper-bound by run_started_ts — fetching happens after
            # run_started_utc is recorded, so fetch_ts is always slightly after it.
            # Just check it's newer than the previous run.
            return True if prev_run_ts is None else fetch_ts > prev_run_ts
        except Exception:
            pass
    # Fall back to published_ts if no fetch timestamp
    published_ts = article.get("published_ts")
    if published_ts is None:
        return False
    try:
        published_ts = float(published_ts)
    except Exception:
        return False
    if published_ts > run_started_ts:
        return False
    return True if prev_run_ts is None else published_ts > prev_run_ts


# ── RSS fetching ───────────────────────────────────────────────────────────────

def load_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def save_seen(path: Path, ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(ids)) + ("\n" if ids else ""))


def load_feed_urls(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def strip_html(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    text = TAG_RE.sub(" ", text)
    text = READ_MORE_RE.sub(" ", text)
    text = BOILERPLATE_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def normalize_feed_url(url: str) -> str:
    url = (url or "").strip()
    return "https://" + url[len("http://"):] if url.startswith("http://") else url


def candidate_feed_urls(url: str) -> list[str]:
    url = (url or "").strip()
    if not url:
        return []
    normalized = normalize_feed_url(url)
    urls: list[str] = [normalized]
    if url not in urls:
        urls.append(url)
    parsed = urlparse(normalized)
    if parsed.scheme == "https":
        http_fallback = "http://" + normalized[len("https://"):]
        if http_fallback not in urls:
            urls.append(http_fallback)
    return urls


def choose_content(entry: Any) -> str:
    if "content" in entry and entry.content:
        first = entry.content[0]
        return first.get("value", "") if isinstance(first, dict) else getattr(first, "value", "")
    return entry.get("summary", "") or entry.get("description", "") or ""


def extract_published(entry: Any) -> str:
    for key in ("published", "updated", "created"):
        if entry.get(key):
            return str(entry.get(key))
    return ""


def extract_published_ts(entry: Any, published_text: str) -> float | None:
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        val = getattr(entry, attr, None)
        if val:
            try:
                return float(calendar.timegm(val))
            except Exception:
                pass
    return parse_utc_string(published_text)


def make_rss_article(entry: Any, feed_url: str, max_content_chars: int) -> dict[str, Any]:
    raw_id = entry.get("id") or entry.get("link") or entry.get("title") or ""
    title = (entry.get("title") or "").strip()
    link = (entry.get("link") or "").strip()
    content = strip_html(choose_content(entry))[:max_content_chars]
    published = extract_published(entry)
    published_ts = extract_published_ts(entry, published)
    normalized_title = normalize_title(title)
    normalized_link = normalize_link(link)
    article_id = hash_text(raw_id or normalized_link or normalized_title, 24)
    content_hash = hash_text(f"{normalized_title}\n{normalized_link}\n{content[:2000]}", 24)
    return {
        "id": article_id,
        "source_id": str(raw_id)[:500],
        "content_hash": content_hash,
        "title": title,
        "link": link,
        "normalized_link": normalized_link,
        "content": content,
        "published": published,
        "published_ts": published_ts,
        "published_utc": iso_from_ts(published_ts),
        "feed": feed_url,
        "fetched_at": utc_now_iso(),
        "timestamp": time.time(),
    }


def parse_one_feed(raw_url: str) -> dict[str, Any]:
    urls = candidate_feed_urls(raw_url)
    last_error: str | None = None
    for url in urls:
        try:
            parsed = feedparser.parse(url)
            bozo = bool(getattr(parsed, "bozo", False))
            bozo_exc = getattr(parsed, "bozo_exception", None)
            entries = list(getattr(parsed, "entries", []) or [])
            if bozo and not entries:
                last_error = str(bozo_exc or "unknown parse error")
                continue
            return {
                "raw_url": raw_url,
                "used_url": url,
                "entries": entries,
                "bozo": bozo,
                "bozo_exception": str(bozo_exc) if bozo_exc else None,
                "error": None,
            }
        except Exception as e:
            last_error = str(e)
            continue
    return {
        "raw_url": raw_url,
        "used_url": urls[0] if urls else raw_url,
        "entries": [],
        "bozo": False,
        "bozo_exception": None,
        "error": last_error or "unknown fetch error",
    }


def process_feed_result(
    result: dict[str, Any],
    args: argparse.Namespace,
    seen: set[str],
    existing_keys: set[str],
    new_entries: list[dict[str, Any]],
    new_ids: list[str],
    new_published_ids: list[str],
    new_hashes: list[str],
    prev_run_ts: float | None,
    run_started_ts: float,
) -> tuple[bool, bool]:
    used_url = result["used_url"]
    entries = result["entries"]
    bozo = bool(result.get("bozo"))

    if bozo and result.get("bozo_exception"):
        print(f"  WARNING: parse issue for {used_url}: {result['bozo_exception']}")

    if not entries:
        return True, bozo

    print(f"  Fetched {len(entries):3d} entries from {used_url}")

    for entry in entries:
        article = make_rss_article(entry, used_url, args.max_content_chars)
        aid = article["id"]
        akey = (
            article.get("content_hash")
            or article.get("normalized_link")
            or normalize_title(article.get("title", ""))
        )
        if not args.force_refresh and (aid in seen or akey in existing_keys):
            continue
        doi_key = extract_doi(article.get("link", "") or "")
        if not args.force_refresh and doi_key and doi_key in existing_keys:
            continue
        if not article.get("title") and not article.get("content"):
            continue
        new_entries.append(article)
        new_ids.append(aid)
        if should_mark_new_published(article, prev_run_ts, run_started_ts):
            new_published_ids.append(aid)
        new_hashes.append(article["content_hash"])
        seen.add(aid)
        existing_keys.add(akey)
        if doi_key:
            existing_keys.add(doi_key)

    return False, bozo


def fetch_rss(
    args: argparse.Namespace,
    seen: set[str],
    existing_keys: set[str],
    new_entries: list[dict[str, Any]],
    new_ids: list[str],
    new_published_ids: list[str],
    new_hashes: list[str],
    prev_run_ts: float | None,
    run_started_ts: float,
) -> dict[str, Any]:
    """Fetch all RSS/Atom feeds. Returns stats dict."""
    feeds = load_feed_urls(args.feeds)
    failed_feeds: list[dict[str, str]] = []
    empty_feeds = ok_feeds = bozo_feeds = recovered_serially = 0

    print(f"[RSS] Fetching {len(feeds)} feeds with up to {args.max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_map = {executor.submit(parse_one_feed, url): url for url in feeds}
        for future in as_completed(future_map):
            raw_url = future_map[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"  WARNING: worker failed for {raw_url}: {e}")
                failed_feeds.append({"feed": raw_url, "error": str(e)})
                continue
            if result["error"]:
                print(f"  WARNING: failed to fetch {raw_url}: {result['error']}")
                failed_feeds.append({"feed": raw_url, "error": result["error"]})
                continue
            is_empty, is_bozo = process_feed_result(
                result, args, seen, existing_keys,
                new_entries, new_ids, new_published_ids, new_hashes,
                prev_run_ts, run_started_ts,
            )
            empty_feeds += int(is_empty)
            ok_feeds += int(not is_empty)
            bozo_feeds += int(is_bozo)

    # Serial retry for failures
    if failed_feeds:
        print(f"  Retrying {len(failed_feeds)} failed feeds serially...")
        retry_failed = failed_feeds[:]
        failed_feeds = []
        for item in retry_failed:
            raw_url = item["feed"]
            try:
                result = parse_one_feed(raw_url)
            except Exception as e:
                failed_feeds.append({"feed": raw_url, "error": str(e)})
                continue
            if result["error"]:
                failed_feeds.append({"feed": raw_url, "error": result["error"]})
                continue
            recovered_serially += 1
            is_empty, is_bozo = process_feed_result(
                result, args, seen, existing_keys,
                new_entries, new_ids, new_published_ids, new_hashes,
                prev_run_ts, run_started_ts,
            )
            empty_feeds += int(is_empty)
            ok_feeds += int(not is_empty)
            bozo_feeds += int(is_bozo)

    return {
        "feeds": len(feeds),
        "working_count": ok_feeds,
        "empty_count": empty_feeds,
        "parse_warning_count": bozo_feeds,
        "failed_feeds": failed_feeds,
        "failed_count": len(failed_feeds),
        "recovered_serially": recovered_serially,
    }


# ── Crossref supplementation ───────────────────────────────────────────────────

def load_interest_terms(path: Path) -> list[str]:
    """Load non-comment, non-empty lines from the interest profile as query terms."""
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def extract_crossref_date(item: dict[str, Any]) -> tuple[float | None, str]:
    for key in ("published-print", "published-online", "published", "issued", "created", "deposited", "indexed"):
        part = item.get(key) or {}
        parts = ((part.get("date-parts") or [[]])[0] or [])
        if parts:
            year = int(parts[0])
            month = int(parts[1]) if len(parts) > 1 else 1
            day = int(parts[2]) if len(parts) > 2 else 1
            try:
                dt = datetime(year, month, day, tzinfo=timezone.utc)
                return dt.timestamp(), dt.strftime("%Y-%m-%d")
            except Exception:
                continue
    return None, ""


def build_crossref_article(item: dict[str, Any], query_term: str) -> dict[str, Any]:
    title = str((item.get("title") or [""])[0]).strip()
    abstract = " ".join(
        (item.get("abstract") or "")
        .replace("<jats:p>", " ")
        .replace("</jats:p>", " ")
        .split()
    )
    doi = (item.get("DOI") or "").strip()
    link = f"https://doi.org/{doi}" if doi else ""
    published_ts, published = extract_crossref_date(item)
    raw_id = doi or link or title
    normalized_link = normalize_link(link)
    normalized_title_val = normalize_title(title)
    content = abstract[:MAX_CONTENT_CHARS]
    content_hash = hash_text(f"{normalized_title_val}\n{normalized_link}\n{content[:2000]}", 24)
    journal = str((item.get("container-title") or [""])[0]).strip()
    return {
        "id": hash_text(raw_id, 24),
        "source_id": raw_id[:500],
        "content_hash": content_hash,
        "title": title,
        "link": link,
        "normalized_link": normalized_link,
        "content": content,
        "published": published,
        "published_ts": published_ts,
        "published_utc": iso_from_ts(published_ts),
        "feed": f"crossref:{query_term}",
        "journal": journal,
        "fetched_at": utc_now_iso(),
        "timestamp": time.time(),
    }


def fetch_crossref_page(
    term: str, offset: int, rows: int, from_date: str, timeout_sec: int, mailto: str
) -> list[dict[str, Any]]:
    params = {
        # Use general query (not query.title) for better concept-level recall
        "query": term,
        "filter": f"type:journal-article,from-pub-date:{from_date}",
        "rows": str(rows),
        "offset": str(offset),
        "select": "DOI,title,abstract,container-title,published,published-print,published-online,issued,created,deposited,indexed",
    }
    url = CROSSREF_API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": CROSSREF_USER_AGENT.format(mailto=mailto or "unknown@example.com")},
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return list((((data or {}).get("message") or {}).get("items") or []))


def fetch_crossref(
    args: argparse.Namespace,
    existing_keys: set[str],
    new_entries: list[dict[str, Any]],
    new_ids: list[str],
    new_published_ids: list[str],
    new_hashes: list[str],
    prev_run_ts: float | None,
    run_started_ts: float,
) -> int:
    """Supplement with Crossref results. Returns count of new articles added."""
    terms_path = args.crossref_terms if args.crossref_terms else args.interest_profile
    terms = load_interest_terms(terms_path)
    if not terms:
        print("[Crossref] No query terms found; skipping.")
        return 0
    print(f"[Crossref] Using {len(terms)} terms from {terms_path.name}.")

    from_date = (
        datetime.now(timezone.utc) - timedelta(days=args.crossref_days_back)
    ).strftime("%Y-%m-%d")

    added = 0
    for term in terms:
        fetched_for_term = 0
        offset = 0
        while fetched_for_term < args.crossref_max_per_term:
            rows = min(args.crossref_rows, args.crossref_max_per_term - fetched_for_term)
            try:
                items = fetch_crossref_page(
                    term, offset, rows, from_date, args.timeout_sec, args.crossref_mailto
                )
            except Exception as e:
                print(f"  [Crossref] WARNING: failed to fetch term '{term}': {e}")
                break
            if not items:
                break
            for item in items:
                article = build_crossref_article(item, term)
                if not article.get("title"):
                    continue
                akey = (
                    article.get("content_hash")
                    or article.get("normalized_link")
                    or normalize_title(article.get("title", ""))
                )
                if akey in existing_keys:
                    continue
                doi_key = extract_doi(article.get("link", "") or "")
                if doi_key and doi_key in existing_keys:
                    continue
                existing_keys.add(akey)
                if doi_key:
                    existing_keys.add(doi_key)
                new_entries.append(article)
                new_ids.append(article["id"])
                new_hashes.append(article["content_hash"])
                if should_mark_new_published(article, prev_run_ts, run_started_ts):
                    new_published_ids.append(article["id"])
                added += 1
            fetched_for_term += len(items)
            offset += len(items)
            if len(items) < rows:
                break
            time.sleep(0.2)

    print(f"[Crossref] Added {added} new records from {len(terms)} terms.")
    return added


# ── High-volume sources (bioRxiv + Nature via Crossref) ────────────────────────

def _api_get(url: str, mailto: str, timeout: int = 60, retries: int = 3) -> dict:
    """HTTP GET with retry and backoff for Crossref cursor-paginated queries."""
    agent = CROSSREF_USER_AGENT.format(mailto=mailto or "unknown@example.com")
    req = urllib.request.Request(url, headers={"User-Agent": agent, "Accept": "application/json"})
    delay = 5.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"    API error (attempt {attempt+1}/{retries}): {e} — retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError("API request failed")


def _load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _fetch_crossref_cursor_pages(
    filter_str: str, mailto: str, max_results: int = 20000,
) -> list[dict[str, Any]]:
    """Cursor-paginated Crossref fetch for large result sets (bioRxiv, ISSN)."""
    items_all: list[dict[str, Any]] = []
    cursor = "*"
    fetched = 0
    while fetched < max_results:
        rows = min(CURSOR_PAGE_SIZE, max_results - fetched)
        params = {
            "filter": filter_str,
            "rows": str(rows),
            "cursor": cursor,
            "select": "DOI,title,abstract,container-title,published,published-print,published-online,issued,posted,indexed",
            "mailto": mailto or "unknown@example.com",
        }
        url = CROSSREF_API_URL + "?" + urllib.parse.urlencode(params)
        try:
            data = _api_get(url, mailto)
        except Exception as e:
            print(f"    Crossref cursor error: {e}")
            break
        message = data.get("message", {})
        items = message.get("items", [])
        if not items:
            break
        items_all.extend(items)
        fetched += len(items)
        print(f"    Fetched {fetched} so far...", end="\r", flush=True)
        next_cursor = message.get("next-cursor")
        if not next_cursor or len(items) < rows:
            break
        cursor = next_cursor
        time.sleep(CURSOR_REQUEST_DELAY)
    print()  # newline after progress
    return items_all


def _add_article_if_new(
    article: dict[str, Any],
    existing_keys: set[str],
    new_entries: list[dict[str, Any]],
    new_ids: list[str],
    new_published_ids: list[str],
    new_hashes: list[str],
    prev_run_ts: float | None,
    run_started_ts: float,
) -> bool:
    """Dedup-check and append an article. Returns True if added."""
    if not article.get("title"):
        return False
    akey = (
        article.get("content_hash")
        or article.get("normalized_link")
        or normalize_title(article.get("title", ""))
    )
    if akey in existing_keys:
        return False
    # Cross-source DOI dedup: catches the same paper fetched via
    # Crossref (doi.org/10.1128/...) and RSS (journals.asm.org/doi/...)
    doi_key = extract_doi(article.get("link", "") or article.get("url", "") or "")
    if doi_key and doi_key in existing_keys:
        return False
    existing_keys.add(akey)
    if doi_key:
        existing_keys.add(doi_key)
    new_entries.append(article)
    new_ids.append(article["id"])
    new_hashes.append(article["content_hash"])
    if should_mark_new_published(article, prev_run_ts, run_started_ts):
        new_published_ids.append(article["id"])
    return True


def _parse_nature_journals(path: Path) -> list[tuple[str, str]]:
    journals = []
    for line in _load_lines(path):
        if "|" in line:
            parts = line.split("|", 1)
            name, issn = parts[0].strip(), parts[1].strip()
            if name and issn:
                journals.append((name, issn))
        else:
            journals.append(("", line.strip()))
    return journals


def fetch_biorxiv(
    args: argparse.Namespace,
    from_date: str,
    existing_keys: set[str],
    new_entries: list[dict[str, Any]],
    new_ids: list[str],
    new_published_ids: list[str],
    new_hashes: list[str],
    prev_run_ts: float | None,
    run_started_ts: float,
) -> int:
    """Fetch bioRxiv preprints via Crossref DOI prefix with keyword filtering."""
    term_stems = [t.lower().strip() for t in _load_lines(args.biorxiv_terms) if t.strip()]
    print(f"[bioRxiv/Crossref] Fetching preprints from {from_date} (both DOI prefixes)...")
    if term_stems:
        print(f"  Keyword filter: {len(term_stems)} terms from {args.biorxiv_terms.name}")
    else:
        print("  WARNING: No biorxiv_terms — fetching ALL preprints (may be very large)")

    items: list[dict[str, Any]] = []
    for prefix in BIORXIV_PREFIXES:
        filter_str = f"prefix:{prefix},type:posted-content,from-pub-date:{from_date}"
        prefix_items = _fetch_crossref_cursor_pages(filter_str, args.crossref_mailto)
        print(f"  Prefix {prefix}: {len(prefix_items)} preprints")
        items.extend(prefix_items)
    print(f"  Total preprints from Crossref: {len(items)}")

    added = filtered_out = 0
    for item in items:
        title = str((item.get("title") or [""])[0]).strip()
        abstract = " ".join((item.get("abstract") or "").replace("<jats:p>", " ").replace("</jats:p>", " ").split())
        text = (title + " " + abstract).lower()
        if term_stems and not any(t in text for t in term_stems):
            filtered_out += 1
            continue
        article = build_crossref_article(item, "crossref-biorxiv")
        if _add_article_if_new(article, existing_keys, new_entries, new_ids,
                               new_published_ids, new_hashes, prev_run_ts, run_started_ts):
            added += 1
    print(f"  Filtered by keyword: {filtered_out} | New articles added: {added}")
    return added


def fetch_nature_journals(
    args: argparse.Namespace,
    from_date: str,
    existing_keys: set[str],
    new_entries: list[dict[str, Any]],
    new_ids: list[str],
    new_published_ids: list[str],
    new_hashes: list[str],
    prev_run_ts: float | None,
    run_started_ts: float,
) -> int:
    """Fetch Nature family journals via Crossref ISSN queries."""
    journals = _parse_nature_journals(args.nature_journals)
    if not journals:
        print("[Nature/Crossref] No journals configured; skipping.")
        return 0
    print(f"[Nature/Crossref] Fetching {len(journals)} journals from {from_date}...")
    total_added = 0
    for journal_name, issn in journals:
        label = journal_name or issn
        print(f"  {label} (ISSN {issn})")
        filter_str = f"issn:{issn},type:journal-article,from-pub-date:{from_date}"
        items = _fetch_crossref_cursor_pages(filter_str, args.crossref_mailto)
        added = 0
        for item in items:
            article = build_crossref_article(item, f"crossref-issn:{journal_name or issn}")
            if _add_article_if_new(article, existing_keys, new_entries, new_ids,
                                   new_published_ids, new_hashes, prev_run_ts, run_started_ts):
                added += 1
        print(f"    → {len(items)} fetched, {added} new")
        total_added += added
    return total_added


def fetch_asm(
    args: argparse.Namespace,
    from_date: str,
    existing_keys: set[str],
    new_entries: list[dict[str, Any]],
    new_ids: list[str],
    new_published_ids: list[str],
    new_hashes: list[str],
    prev_run_ts: float | None,
    run_started_ts: float,
) -> int:
    """Fetch ASM Press journals via Crossref DOI prefix (10.1128).

    ASM's RSS feeds are blocked by Cloudflare bot protection and their
    Crossref ISSN indexing lags by days/weeks, so we query by DOI prefix
    instead.  All ASM journals (mBio, AEM, JB, mSphere, mSystems, etc.)
    share the 10.1128 prefix.  No keyword filtering is applied since all
    ASM content is microbiology — the downstream semantic filter and LLM
    ranker handle precision.
    """
    print(f"[ASM/Crossref] Fetching ASM journals from {from_date} (prefix 10.1128)...")
    items: list[dict[str, Any]] = []
    for prefix in ASM_PREFIXES:
        filter_str = f"prefix:{prefix},type:journal-article,from-pub-date:{from_date}"
        prefix_items = _fetch_crossref_cursor_pages(filter_str, args.crossref_mailto)
        print(f"  Prefix {prefix}: {len(prefix_items)} articles")
        items.extend(prefix_items)
    print(f"  Total ASM articles from Crossref: {len(items)}")

    added = 0
    for item in items:
        article = build_crossref_article(item, "crossref-asm")
        if _add_article_if_new(article, existing_keys, new_entries, new_ids,
                               new_published_ids, new_hashes, prev_run_ts, run_started_ts):
            added += 1
    print(f"  New articles added: {added}")
    return added


# ── Deduplication ──────────────────────────────────────────────────────────────

def dedupe_articles(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_hashes: set[str] = set()
    seen_titles: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for rec in sorted(records, key=lambda x: x.get("timestamp", 0), reverse=True):
        # Check both content hash AND normalised title to catch near-duplicates
        # (same article fetched twice with slightly different content)
        chash = rec.get("content_hash") or rec.get("normalized_link") or rec.get("id")
        title = normalize_title(rec.get("title", ""))
        if chash and chash in seen_hashes:
            continue
        if title and title in seen_titles:
            continue
        if chash:
            seen_hashes.add(chash)
        if title:
            seen_titles.add(title)
        deduped.append(rec)
    deduped.reverse()
    return deduped


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_project_dirs()
    args = parse_args()
    socket.setdefaulttimeout(args.timeout_sec)

    prior_state = load_json(args.fetch_state, {})
    prev_run_utc = args.last_run_utc.strip() or str(prior_state.get("last_run_utc", "")).strip()
    run_started_utc = args.run_started_utc.strip() or utc_now_iso()
    prev_run_ts = parse_utc_string(prev_run_utc)
    run_started_ts = parse_utc_string(run_started_utc) or time.time()

    seen = set() if args.force_refresh else load_seen(args.seen_ids)
    raw = load_json(args.raw_cache, [])
    existing_keys: set[str] = set()
    for rec in raw:
        # Primary key: content_hash (or fallback to normalized_link / title)
        pkey = rec.get("content_hash") or rec.get("normalized_link") or normalize_title(rec.get("title", ""))
        if pkey:
            existing_keys.add(pkey)
        # DOI key: enables cross-source dedup (same paper via Crossref vs RSS)
        for field in ("link", "url", "normalized_link"):
            doi_key = extract_doi(rec.get(field, "") or "")
            if doi_key:
                existing_keys.add(doi_key)

    new_entries: list[dict[str, Any]] = []
    new_ids: list[str] = []
    new_published_ids: list[str] = []
    new_hashes: list[str] = []

    # 1. RSS fetch
    rss_stats = fetch_rss(
        args, seen, existing_keys,
        new_entries, new_ids, new_published_ids, new_hashes,
        prev_run_ts, run_started_ts,
    )
    rss_added = len(new_entries)

    # 2. Crossref supplement
    crossref_added = 0
    if not args.skip_crossref:
        crossref_added = fetch_crossref(
            args, existing_keys,
            new_entries, new_ids, new_published_ids, new_hashes,
            prev_run_ts, run_started_ts,
        )

    # 3. High-volume sources (bioRxiv + Nature via Crossref cursor pagination)
    #    Use 1-day buffer before prev_run_ts to catch late-indexed articles.
    hv_from_dt = (
        datetime.fromtimestamp(prev_run_ts, tz=timezone.utc) - timedelta(days=1)
        if prev_run_ts
        else datetime.now(timezone.utc) - timedelta(days=2)
    )
    hv_from_date = hv_from_dt.strftime("%Y-%m-%d")

    biorxiv_added = 0
    if not args.skip_biorxiv:
        biorxiv_added = fetch_biorxiv(
            args, hv_from_date, existing_keys,
            new_entries, new_ids, new_published_ids, new_hashes,
            prev_run_ts, run_started_ts,
        )

    nature_added = 0
    if not args.skip_nature:
        nature_added = fetch_nature_journals(
            args, hv_from_date, existing_keys,
            new_entries, new_ids, new_published_ids, new_hashes,
            prev_run_ts, run_started_ts,
        )

    # ASM Press (DOI prefix 10.1128) — uses hv_from_date (prev_run - 1 day)
    # like bioRxiv and Nature.  This keeps the fetch window inside the prune
    # window so articles don't cycle through prune → refetch → digest.
    # For backfills, use MAX_AGE_DAYS=45 which both prevents pruning and
    # pushes hv_from_date far enough back to catch older articles.
    asm_added = 0
    if not args.skip_asm:
        asm_added = fetch_asm(
            args, hv_from_date, existing_keys,
            new_entries, new_ids, new_published_ids, new_hashes,
            prev_run_ts, run_started_ts,
        )

    # 4. Persist
    combined = dedupe_articles(raw + new_entries)
    save_json(args.raw_cache, combined)
    save_json(args.new_ids, new_ids)
    save_json(args.new_published_ids, new_published_ids)
    save_json(args.new_hashes, new_hashes)
    save_seen(args.seen_ids, seen)
    save_json(
        args.fetch_state,
        {
            "previous_run_utc": prev_run_utc,
            "run_started_utc": run_started_utc,
            "last_run_utc": run_started_utc,
            "rss_new_count": rss_added,
            "crossref_added_count": crossref_added,
            "hv_biorxiv_added": biorxiv_added,
            "hv_nature_added": nature_added,
            "hv_asm_added": asm_added,
            "new_count": len(new_entries),
            "new_published_count": len(new_published_ids),
            "total_cached": len(combined),
            "max_workers": args.max_workers,
            "timeout_sec": args.timeout_sec,
            **rss_stats,
        },
    )

    print(
        f"\nFetch complete: {rss_added} RSS, {crossref_added} Crossref, "
        f"{biorxiv_added} bioRxiv, {nature_added} Nature, {asm_added} ASM. "
        f"{len(new_published_ids)} published since last run. "
        f"Cache total: {len(combined)}."
    )


if __name__ == "__main__":
    main()
