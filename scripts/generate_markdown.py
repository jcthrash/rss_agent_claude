#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from common_paths import MARKDOWN_DIR as OUT_MD, HTML_DIR as OUT_HTML, CSV_DIR as OUT_CSV, NEW_PUBLISHED_IDS_JSON

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANKED_PATH = PROJECT_ROOT / 'data' / 'cache' / 'ranked_articles.json'

# ── Digest dedup: track articles already shown in a previous digest ────────────
# This is the authoritative "have I seen this?" gate.  Unlike new_published_ids
# (which depends on fetch-time bookkeeping), digested_ids tracks what actually
# appeared in an output file.  Robust against backfills, prune/refetch cycles,
# and any upstream pipeline quirks.
DIGESTED_IDS_PATH = PROJECT_ROOT / 'data' / 'cache' / 'digested_ids.json'
DIGESTED_MAX_AGE_DAYS = 365   # auto-prune entries older than this

# ── Score-based tier defaults ──────────────────────────────────────────────────
DEFAULT_TOP_PICKS_SCORE = 0.75
DEFAULT_MIN_SCORE       = 0.45

HTML_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Arial, sans-serif; line-height: 1.55; margin: 32px auto; max-width: 1100px; padding: 0 24px 48px; color: #1f2937; background: #ffffff; }
h1, h2 { color: #111827; }
h1 { margin-bottom: 0.25rem; }
p.muted { color: #6b7280; }
ol { padding-left: 1.25rem; }
li.article { margin: 0 0 1rem 0; padding: 0.85rem 1rem; border: 1px solid #e5e7eb; border-radius: 10px; background: #ffffff; }
.article-title { font-size: 1rem; font-weight: 700; }
.meta { margin: 0.35rem 0; color: #374151; font-size: 0.94rem; }
.summary { margin: 0.35rem 0; }
""".strip()


def load_json(path: Path, default: Any):
    return default if not path.exists() else json.loads(path.read_text(encoding='utf-8'))


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def ensure_dirs() -> None:
    OUT_MD.mkdir(parents=True, exist_ok=True)
    OUT_HTML.mkdir(parents=True, exist_ok=True)
    OUT_CSV.mkdir(parents=True, exist_ok=True)


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def short_summary(article: Dict[str, Any], n: int = 320) -> str:
    s = ' '.join(((article.get('summary') or article.get('abstract') or article.get('content') or '').strip()).split())
    return s[:n] + ('...' if len(s) > n else '')


def article_sort_score(article: Dict[str, Any]) -> float:
    return safe_float(article.get('score', article.get('llm_score', article.get('semantic_score', 0.0))))


def article_id(article: Dict[str, Any]) -> str:
    """Return the best available unique identifier for dedup."""
    return article.get('content_hash') or article.get('id') or ''


# ── DOI extraction for cross-source dedup ──────────────────────────────────────
_DOI_RE = re.compile(r'\b(10\.\d{4,9}/[^\s?#<>"\'\]]+)', re.IGNORECASE)

# Publisher URL → DOI fallbacks (when the DOI isn't literally in the URL).
# Each tuple is (compiled regex matching the URL, DOI prefix to prepend to the
# captured group).  Add new publishers here as needed — no logic changes required.
_PUBLISHER_DOI_PATTERNS = [
    # nature.com/articles/<id> → 10.1038/<id>
    (re.compile(r'https?://(?:www\.)?nature\.com/articles/([a-z0-9._-]+)', re.IGNORECASE),
     '10.1038/'),
]


def _extract_doi(url: str) -> str:
    if not url:
        return ''
    m = _DOI_RE.search(url)
    if not m:
        # Check publisher-specific URL patterns
        for pattern, prefix in _PUBLISHER_DOI_PATTERNS:
            pm = pattern.match(url)
            if pm:
                return 'doi:' + prefix + pm.group(1).lower()
        return ''
    doi = m.group(1).lower().rstrip('.,;:)')
    doi = re.sub(r'/(full|abstract|epdf|pdf)$', '', doi)
    doi = re.sub(r'v\d+$', '', doi)
    doi = re.sub(r'/\d+$', '', doi)
    return 'doi:' + doi


def article_doi(article: Dict[str, Any]) -> str:
    """Extract DOI from an article's URL fields."""
    for field in ('link', 'url', 'normalized_link'):
        doi = _extract_doi(article.get(field, '') or '')
        if doi:
            return doi
    return ''


# ── Digested-ID tracking ──────────────────────────────────────────────────────

def load_digested_ids(path: Path) -> dict[str, float]:
    """Load {article_id: epoch_timestamp} of previously digested articles."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            return data
        # Migrate from old list format if needed
        if isinstance(data, list):
            now = time.time()
            return {str(aid): now for aid in data}
    except Exception:
        pass
    return {}


def prune_digested_ids(digested: dict[str, float], max_age_days: int) -> dict[str, float]:
    """Remove entries older than max_age_days."""
    cutoff = time.time() - (max_age_days * 86400)
    return {aid: ts for aid, ts in digested.items() if ts > cutoff}


def save_digested_ids(path: Path, digested: dict[str, float]) -> None:
    save_json(path, digested)


# ── Formatting ─────────────────────────────────────────────────────────────────

def format_article_md(article: Dict[str, Any]) -> str:
    title = article.get('title', 'Untitled')
    link = article.get('link') or article.get('url')
    score = safe_float(article.get('score', article.get('llm_score', 0.0)))
    feed = article.get('feed', '')
    published = article.get('published_utc') or article.get('published', '')
    summary = short_summary(article)
    lines = [f'- **[{title}]({link})**' if link else f'- **{title}**']
    meta = f'  - Relevance: {score:.3f}'
    if published:
        meta += f' | Published: {published}'
    if feed:
        meta += f' | Feed: {feed}'
    lines.append(meta)
    if summary:
        lines.append(f'  - {summary}')
    return '\n'.join(lines)


def format_article_html(article: Dict[str, Any]) -> str:
    title = html.escape(article.get('title', 'Untitled'))
    link = article.get('link') or article.get('url')
    score = safe_float(article.get('score', article.get('llm_score', 0.0)))
    feed = html.escape(article.get('feed', ''))
    published = html.escape(article.get('published_utc') or article.get('published', ''))
    summary = html.escape(short_summary(article))
    title_html = f'<a href="{html.escape(link)}">{title}</a>' if link else title
    pieces = ['<li class="article">', f'<div class="article-title">{title_html}</div>', f'<div class="meta"><strong>Relevance:</strong> {score:.3f}</div>']
    if published or feed:
        pieces.append(f'<div class="meta"><strong>Published:</strong> {published or "n/a"} &nbsp; <strong>Feed:</strong> {feed or "n/a"}</div>')
    if summary:
        pieces.append(f'<div class="summary">{summary}</div>')
    pieces.append('</li>')
    return '\n'.join(pieces)


def write_csv(path: Path, articles: List[Dict[str, Any]]) -> None:
    fields = ['title', 'link', 'published', 'published_utc', 'feed', 'semantic_score', 'profile_similarity', 'library_similarity', 'llm_score', 'score']
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for a in articles:
            writer.writerow({k: a.get(k, '') for k in fields})


def render_markdown(title: str, top_picks: List[Dict[str, Any]], scan_pool: List[Dict[str, Any]]) -> str:
    md = [f'# {title}', '', f'Top Picks: {len(top_picks)}', f'Also Worth Scanning: {len(scan_pool)}', '']
    if top_picks:
        md += ['## Top Picks', ''] + [format_article_md(a) for a in top_picks] + ['']
    if scan_pool:
        md += ['## Also Worth Scanning', ''] + [format_article_md(a) for a in scan_pool] + ['']
    return '\n'.join(md)


def render_html(title: str, top_picks: List[Dict[str, Any]], scan_pool: List[Dict[str, Any]]) -> str:
    parts = ['<html><head><meta charset="utf-8">', f'<title>{html.escape(title)}</title>', f'<style>{HTML_CSS}</style>', '</head><body>', f'<h1>{html.escape(title)}</h1>', f'<p class="muted">Top Picks: {len(top_picks)} | Also Worth Scanning: {len(scan_pool)}</p>']
    if top_picks:
        parts += ['<h2>Top Picks</h2>', '<ol>'] + [format_article_html(a) for a in top_picks] + ['</ol>']
    if scan_pool:
        parts += ['<h2>Also Worth Scanning</h2>', '<ol>'] + [format_article_html(a) for a in scan_pool] + ['</ol>']
    return '\n'.join(parts + ['</body></html>'])


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dirs()
    articles = load_json(RANKED_PATH, [])
    if not articles:
        print('No ranked articles found; no digest generated.')
        return

    # ── Load digested-article tracking ─────────────────────────────────────────
    reset_digested = os.environ.get('RESET_DIGESTED', '0') == '1'
    if reset_digested:
        digested = {}
        print('RESET_DIGESTED=1 — cleared digested article history.')
    else:
        digested = load_digested_ids(DIGESTED_IDS_PATH)
        digested = prune_digested_ids(digested, DIGESTED_MAX_AGE_DAYS)

    digested_id_set = set(digested.keys())
    # digested_ids stores both content_hash keys AND doi: keys,
    # so DOI-based lookups work automatically via the same set.

    # ── Filter to new-to-digest articles ───────────────────────────────────────
    # Primary gate: exclude anything already shown in a previous digest,
    # checking both content_hash and DOI (catches cross-source duplicates
    # where the same paper was fetched via Crossref and RSS with different URLs).
    # Secondary gate: intersect with new_published_ids when available.
    published_new_ids = set(load_json(NEW_PUBLISHED_IDS_JSON, []))

    candidates = articles
    if published_new_ids:
        candidates = [a for a in candidates if a.get('id') in published_new_ids]

    # Exclude previously digested articles (by content_hash OR DOI)
    before_dedup = len(candidates)

    def _is_digested(a: Dict[str, Any]) -> bool:
        if article_id(a) in digested_id_set:
            return True
        doi = article_doi(a)
        if doi and doi in digested_id_set:
            return True
        return False

    candidates = [a for a in candidates if not _is_digested(a)]

    # Also deduplicate within the current batch by DOI (same-day cross-source dupes)
    seen_dois: set[str] = set()
    deduped: list[Dict[str, Any]] = []
    for a in candidates:
        doi = article_doi(a)
        if doi and doi in seen_dois:
            continue
        if doi:
            seen_dois.add(doi)
        deduped.append(a)
    candidates = deduped

    if not candidates:
        # Check if this is a fallback run (RANK_ONLY_NEW=0)
        if any(a.get('used_fallback') for a in articles):
            print('Fallback run: generating digest from full ranked cache (excluding previously digested).')
            candidates = [a for a in articles if not _is_digested(a)]
        elif not published_new_ids:
            print('No new_published_ids found; using all ranked articles (excluding previously digested).')
            candidates = [a for a in articles if not _is_digested(a)]

    if not candidates:
        print('No new articles to digest (all candidates previously digested).')
        return

    if before_dedup > len(candidates) and before_dedup > 0:
        print(f'Excluded {before_dedup - len(candidates)} previously digested / duplicate articles.')

    candidates = sorted(candidates, key=article_sort_score, reverse=True)

    # ── Score-based tiers ──────────────────────────────────────────────────────
    top_picks_score = float(os.environ.get('TOP_PICKS_SCORE') or DEFAULT_TOP_PICKS_SCORE)
    min_score       = float(os.environ.get('MIN_SCORE')       or DEFAULT_MIN_SCORE)

    top_picks = [a for a in candidates if article_sort_score(a) >= top_picks_score]
    scan_pool = [a for a in candidates if min_score <= article_sort_score(a) < top_picks_score]
    export_articles = top_picks + scan_pool

    if not export_articles:
        print(f'No articles scored above MIN_SCORE={min_score:.2f}; no digest generated.')
        return

    # ── Generate outputs ───────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    title = f"New Since Last Run — {datetime.now().strftime('%Y-%m-%d')}"
    md_text = render_markdown(title, top_picks, scan_pool)
    html_text = render_html(title, top_picks, scan_pool)

    for stem, text in [(OUT_MD / 'curated_today.md', md_text), (OUT_MD / f'curated_today_{timestamp}.md', md_text)]:
        stem.write_text(text, encoding='utf-8')
    for stem, text in [(OUT_HTML / 'curated_today.html', html_text), (OUT_HTML / f'curated_today_{timestamp}.html', html_text)]:
        stem.write_text(text, encoding='utf-8')
    write_csv(OUT_CSV / 'curated_today.csv', export_articles)
    write_csv(OUT_CSV / f'curated_today_{timestamp}.csv', export_articles)

    # ── Update digested tracking ───────────────────────────────────────────────
    # Store both content_hash and DOI for each article so that cross-source
    # duplicates (same DOI, different URL) are caught on future runs.
    now = time.time()
    for a in export_articles:
        aid = article_id(a)
        if aid:
            digested[aid] = now
        doi = article_doi(a)
        if doi:
            digested[doi] = now
    save_digested_ids(DIGESTED_IDS_PATH, digested)

    print(f'Top Picks (>={top_picks_score:.2f}): {len(top_picks)} | '
          f'Also Worth Scanning (>={min_score:.2f}): {len(scan_pool)} | '
          f'Total in digest: {len(export_articles)} | '
          f'Digested history: {len(digested)} articles tracked')
    print(f"Wrote digest to {OUT_MD / 'curated_today.md'}")
    print(f"Wrote HTML to {OUT_HTML / 'curated_today.html'}")
    print(f"Wrote CSV to {OUT_CSV / 'curated_today.csv'}")


if __name__ == '__main__':
    main()
