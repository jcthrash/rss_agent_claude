#!/usr/bin/env python3
"""
eval_pipeline.py

Precision/recall diagnostic tool for the RSS pipeline.

Fetches your Bluesky posts, extracts article URLs and titles as ground truth,
then matches them against the pipeline's raw cache, score caches, and ranked
output to show:

  - True positives:  articles you posted that the pipeline also surfaced
    • current_run:  in the most recent ranked output
    • prior_run:    scored above MIN_SCORE in a previous run (cached LLM score)
  - False negatives: articles you posted that the pipeline never surfaced
  - False positives: articles in the pipeline output that you didn't post

For each false negative, shows exactly where it was lost:
  - NOT_IN_CACHE    → the article was never fetched (feed/Crossref issue)
  - AGE_EXCLUDED    → fetched but excluded by age/newness gate before scoring
  - FILTERED_OUT    → fetched + scored but semantic score below threshold
  - RANKED_OUT      → passed filter, LLM-scored, but final score below MIN_SCORE

Usage:
    python scripts/eval_pipeline.py --days-back 3
    python scripts/eval_pipeline.py --days-back 7 --min-score 0.45
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────────
DEFAULT_RAW_CACHE      = PROJECT_ROOT / "data/cache/raw_feeds.json"
DEFAULT_FILTERED       = PROJECT_ROOT / "data/cache/filtered_articles.json"
DEFAULT_RANKED         = PROJECT_ROOT / "data/cache/ranked_articles.json"
DEFAULT_LLM_CACHE      = PROJECT_ROOT / "data/cache/llm_scores.json"
DEFAULT_SEMANTIC_CACHE  = PROJECT_ROOT / "data/cache/semantic_scores.json"
DEFAULT_NOVELTY_CACHE   = PROJECT_ROOT / "data/cache/novelty_scores.json"
DEFAULT_REPORT_OUT     = PROJECT_ROOT / "data/outputs/reports/pipeline_eval.json"

BLUESKY_API = "https://public.api.bsky.app/xrpc"
HANDLE      = os.environ.get("BLUESKY_HANDLE", "")  # Set via env or --handle


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pipeline precision/recall against Bluesky posts")
    p.add_argument("--handle",     default=HANDLE,           help="Bluesky handle to fetch posts from")
    p.add_argument("--days-back",  type=int, default=30,     help="How many days back to fetch (default: 30)")
    p.add_argument("--raw",        type=Path, default=DEFAULT_RAW_CACHE)
    p.add_argument("--filtered",   type=Path, default=DEFAULT_FILTERED)
    p.add_argument("--ranked",     type=Path, default=DEFAULT_RANKED)
    p.add_argument("--llm-cache",  type=Path, default=DEFAULT_LLM_CACHE)
    p.add_argument("--semantic-cache", type=Path, default=DEFAULT_SEMANTIC_CACHE)
    p.add_argument("--novelty-cache",  type=Path, default=DEFAULT_NOVELTY_CACHE)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--min-score",  type=float, default=0.45,
                   help="Final score threshold — articles at or above this are considered "
                        "successfully surfaced (default: 0.45, must match MIN_SCORE in run_pipeline.sh)")
    p.add_argument("--threshold",  type=float, default=0.25,
                   help="Semantic filter threshold used by the pipeline (default: 0.25)")
    return p.parse_args()


# ── Bluesky fetching ───────────────────────────────────────────────────────────

def bsky_get(endpoint: str, params: dict) -> dict:
    url = f"{BLUESKY_API}/{endpoint}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_author_posts(handle: str, days_back: int) -> list[dict]:
    """Fetch all posts from a public Bluesky account within the last `days_back` days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    posts = []
    cursor = None

    print(f"Fetching Bluesky posts for @{handle} (last {days_back} days)...")

    while True:
        params: dict[str, Any] = {"actor": handle, "limit": 100, "filter": "posts_no_replies"}
        if cursor:
            params["cursor"] = cursor

        try:
            data = bsky_get("app.bsky.feed.getAuthorFeed", params)
        except Exception as e:
            print(f"  WARNING: Bluesky API error: {e}")
            break

        feed = data.get("feed", [])
        if not feed:
            break

        for item in feed:
            post = item.get("post", {})
            record = post.get("record", {})
            created_at = record.get("createdAt", "")
            try:
                post_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except Exception:
                continue

            if post_time < cutoff:
                return posts

            posts.append({
                "uri":        post.get("uri", ""),
                "created_at": created_at,
                "text":       record.get("text", ""),
                "embed":      record.get("embed", {}),
                "facets":     record.get("facets", []),
            })

        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(0.3)

    return posts


# ── URL extraction from Bluesky posts ─────────────────────────────────────────

URL_RE = re.compile(r'https?://[^\s\]>)"\']+')


def extract_urls_from_post(post: dict) -> list[str]:
    """Extract all URLs from a Bluesky post: text, facets, and embed card."""
    urls = []
    for facet in post.get("facets", []):
        for feature in facet.get("features", []):
            if feature.get("$type") == "app.bsky.richtext.facet#link":
                uri = feature.get("uri", "")
                if uri:
                    urls.append(uri)

    embed = post.get("embed", {})
    embed_type = embed.get("$type", "")
    if "external" in embed_type:
        external = embed.get("external", {})
        uri = external.get("uri", "")
        if uri:
            urls.append(uri)

    text = post.get("text", "")
    for m in URL_RE.finditer(text):
        urls.append(m.group(0).rstrip(".,;)\"'"))

    seen: set[str] = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def extract_title_from_post(post: dict) -> str:
    """Extract the article title from an embed card if present."""
    embed = post.get("embed", {})
    if "external" in embed.get("$type", ""):
        return embed.get("external", {}).get("title", "").strip()
    return ""


def extract_title_from_post_text(text: str) -> str:
    """Extract article title from a Bluesky post body when the embed card
    title is missing.  Strips URLs, truncated URLs, hashtags, and emoji."""
    if not text:
        return ""
    t = text
    t = re.sub(r'\s*#\w+', '', t)
    t = re.sub(r'https?://\S+', '', t)
    t = re.sub(r'\s*\b[\w-]+\.[\w.-]+/\S*(?:[\u2026]|\.\.\.)\S*', '', t)
    t = re.sub(r'\s*\b[\w-]+\.[\w.-]+/\S+', '', t)
    t = re.sub(r'[\U0001F300-\U0001FAFF\U00002600-\U000027BF]+', '', t)
    return re.sub(r'\s+', ' ', t).strip()


def is_article_url(url: str) -> bool:
    skip_domains = {
        "bsky.app", "twitter.com", "x.com", "facebook.com",
        "instagram.com", "linkedin.com", "youtube.com", "youtu.be",
    }
    try:
        host = urllib.parse.urlparse(url).netloc.lower().replace("www.", "")
        return not any(host == d or host.endswith("." + d) for d in skip_domains)
    except Exception:
        return False


def build_ground_truth(posts: list[dict]) -> list[dict]:
    ground_truth = []
    for post in posts:
        urls = [u for u in extract_urls_from_post(post) if is_article_url(u)]
        title = extract_title_from_post(post)
        post_text = post.get("text", "")
        if not title and post_text:
            title = extract_title_from_post_text(post_text)
        for url in urls:
            ground_truth.append({
                "url":        url,
                "title":      title,
                "post_text":  post_text[:200],
                "created_at": post.get("created_at", ""),
            })
    return ground_truth


# ── URL normalisation for matching ────────────────────────────────────────────

def normalise_url(url: str) -> str:
    url = (url or "").strip().lower()
    url = re.sub(r"^https?://(www\.)?", "", url)
    url = url.split("?")[0].split("#")[0].rstrip("/")
    url = re.sub(r"/doi/(full|abs|pdf|epdf)/", "/doi/", url)
    url = re.sub(r"\.(full|abstract|pdf)$", "", url)
    return url


# ── Identifier extraction (DOI / Nature article ID / OUP slug) ────────────────

DOI_RE       = re.compile(r'\b(10\.\d{4,9}/[^\s?#<>"\'\]]+)', re.I)
NATURE_ID_RE = re.compile(r'/articles/(s\d+-\d+-\d+-[a-z0-9]+)', re.I)
SLUG_RE      = re.compile(r'/([a-z]{3,6}\d{2,6}[a-z]?)(?=/|$|\?)', re.I)


def _clean_doi(doi: str) -> str:
    doi = doi.rstrip('.,;:)')
    doi = re.sub(r'/(full|abstract|epdf|pdf)$', '', doi, flags=re.I)
    doi = re.sub(r'v\d+$', '', doi)
    return doi.lower()


def extract_identifiers(url: str) -> set[str]:
    ids: set[str] = set()
    if not url:
        return ids
    u = url.strip()
    for m in DOI_RE.finditer(u):
        doi = _clean_doi(m.group(1))
        ids.add('doi:' + doi)
        shortened = re.sub(r'/\d+$', '', doi)
        if shortened != doi:
            ids.add('doi:' + shortened)
    for m in NATURE_ID_RE.finditer(u):
        ids.add('nature:' + m.group(1).lower())
    for m in SLUG_RE.finditer(u):
        slug = m.group(1).lower()
        if re.search(r'\d', slug) and re.search(r'[a-z]', slug) and 5 <= len(slug) <= 10:
            ids.add('slug:' + slug)
    return ids


# ── Title normalisation and matching ──────────────────────────────────────────

def normalise_title(title: str) -> str:
    return re.sub(r'\s+', ' ', (title or "").strip().lower())


def titles_match(a: str, b: str, threshold: int = 8) -> bool:
    if not a or not b:
        return False
    words_a = set(re.findall(r'[a-z0-9]{4,}', normalise_title(a)))
    words_b = set(re.findall(r'[a-z0-9]{4,}', normalise_title(b)))
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b)
    return overlap >= min(threshold, len(words_a) // 2 + 1)


def find_in_corpus(gt: dict, corpus: list[dict]) -> dict | None:
    """Return the first article in corpus that matches a ground truth entry."""
    gt_url   = normalise_url(gt["url"])
    gt_title = normalise_title(gt["title"])
    gt_ids   = extract_identifiers(gt["url"])

    for article in corpus:
        if gt_ids:
            for field in ("link", "url", "normalized_link", "doi"):
                a_ids = extract_identifiers(article.get(field, "") or "")
                if a_ids and (gt_ids & a_ids):
                    return article
        for field in ("link", "url", "normalized_link"):
            a_url = normalise_url(article.get(field, "") or "")
            if a_url and gt_url and (a_url == gt_url or gt_url in a_url or a_url in gt_url):
                return article
        a_title = normalise_title(article.get("title", ""))
        if gt_title and titles_match(gt_title, a_title):
            return article

    return None


# ── Score reconstruction ──────────────────────────────────────────────────────

def reconstruct_final_score(llm_score: float, novelty_score: float) -> float:
    """Reproduce the final score formula from filter_and_rank.py:
       score = 0.90 * llm + 0.10 * novelty"""
    return 0.90 * llm_score + 0.10 * novelty_score


def score_summary(article: dict) -> dict:
    return {
        "semantic_score":    round(float(article.get("semantic_score",    0.0) or 0.0), 4),
        "library_similarity":round(float(article.get("library_similarity",0.0) or 0.0), 4),
        "profile_similarity":round(float(article.get("profile_similarity",0.0) or 0.0), 4),
        "lexical_score":     round(float(article.get("lexical_score",     0.0) or 0.0), 4),
        "llm_score":         round(float(article.get("llm_score",         0.0) or 0.0), 4)
                             if article.get("llm_score") is not None else None,
        "score":             round(float(article.get("score",             0.0) or 0.0), 4),
        "novelty_score":     round(float(article.get("novelty_score",     0.0) or 0.0), 4),
    }


# ── Main evaluation ────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Load pipeline data ─────────────────────────────────────────────────────
    def load(path: Path) -> list[dict]:
        if not path.exists():
            print(f"  WARNING: {path} not found — treating as empty.")
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []

    def load_dict(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    raw      = load(args.raw)
    filtered = load(args.filtered)
    ranked   = load(args.ranked)
    llm_cache     = load_dict(args.llm_cache)
    semantic_cache = load_dict(args.semantic_cache)
    novelty_cache  = load_dict(args.novelty_cache)

    def article_cache_key(a: dict) -> str:
        return a.get("content_hash") or a.get("id") or ""

    # Build comprehensive ranked lookup — URLs, identifiers, and keys
    ranked_urls: set[str] = set()
    ranked_ids:  set[str] = set()
    ranked_keys: set[str] = set()
    for a in ranked:
        for field in ("link", "url", "normalized_link"):
            u = normalise_url(a.get(field, "") or "")
            if u:
                ranked_urls.add(u)
            ranked_ids |= extract_identifiers(a.get(field, "") or "")
        k = article_cache_key(a)
        if k:
            ranked_keys.add(k)

    print(f"Pipeline data loaded: {len(raw)} raw, {len(filtered)} filtered, "
          f"{len(ranked)} ranked, {len(llm_cache)} llm-scored, "
          f"{len(novelty_cache)} novelty-scored.")

    # ── Fetch Bluesky ground truth ─────────────────────────────────────────────
    if not args.handle:
        raise SystemExit(
            "No Bluesky handle provided. Set BLUESKY_HANDLE in .env or pass --handle yourname.bsky.social"
        )
    posts = fetch_author_posts(args.handle, args.days_back)
    print(f"Fetched {len(posts)} Bluesky posts.")

    ground_truth = build_ground_truth(posts)
    print(f"Extracted {len(ground_truth)} article references from posts.\n")

    if not ground_truth:
        print("No article URLs found in Bluesky posts. Nothing to evaluate.")
        return

    # ── Helper: check if pipeline ever successfully scored an article ──────────
    def check_prior_run_success(cache_key: str) -> tuple[bool, float | None, float]:
        """Check if the pipeline ever LLM-scored this article above min_score.

        Returns (is_prior_tp, llm_score, final_score).
        """
        if not cache_key or cache_key not in llm_cache:
            return False, None, 0.0
        llm = float(llm_cache[cache_key])
        nov_data = novelty_cache.get(cache_key, {})
        novelty = float(nov_data.get("novelty_score", 0.0)) if isinstance(nov_data, dict) else 0.0
        final = reconstruct_final_score(llm, novelty)
        return final >= args.min_score, llm, final

    # ── Match each ground truth article against pipeline stages ───────────────
    true_positives  = []
    false_negatives = []

    for gt in ground_truth:
        ranked_match   = find_in_corpus(gt, ranked)
        filtered_match = find_in_corpus(gt, filtered)
        raw_match      = find_in_corpus(gt, raw)

        gt_url_norm  = normalise_url(gt["url"])
        gt_ids_local = extract_identifiers(gt["url"])

        # ── Check 1: in current run's ranked output? ──────────────────────────
        in_ranked = (
            ranked_match is not None
            or (gt_url_norm and gt_url_norm in ranked_urls)
            or bool(gt_ids_local & ranked_ids)
            or (filtered_match and article_cache_key(filtered_match) in ranked_keys)
        )

        if in_ranked:
            article = ranked_match or filtered_match or {}
            fkey = article_cache_key(article) if article else ""
            llm_score = article.get("llm_score")
            if llm_score is None and fkey and fkey in llm_cache:
                llm_score = float(llm_cache[fkey])
            true_positives.append({
                "ground_truth": gt,
                "run": "current",
                "pipeline_article": {
                    "title": article.get("title"),
                    "link":  article.get("link"),
                    **score_summary(article),
                    "llm_score": llm_score,
                },
            })
            continue

        # ── Check 2: ever successfully scored in a prior run? ─────────────────
        # Look up the article's cache key from whichever corpus match we have
        best_match = filtered_match or raw_match
        prior_key = article_cache_key(best_match) if best_match else ""
        is_prior_tp, prior_llm, prior_final = check_prior_run_success(prior_key)

        if is_prior_tp:
            # Pipeline caught this in a previous run — count as TP
            scores = score_summary(best_match) if best_match else {}
            scores["llm_score"] = prior_llm
            scores["score"] = round(prior_final, 4)
            true_positives.append({
                "ground_truth": gt,
                "run": "prior",
                "pipeline_article": {
                    "title": best_match.get("title") if best_match else None,
                    "link":  best_match.get("link") if best_match else None,
                    **scores,
                },
            })
            continue

        # ── False negative — determine where it dropped out ───────────────────
        if raw_match:
            raw_key = article_cache_key(raw_match)
            cached_sem = semantic_cache.get(raw_key, {})
            was_scored = raw_key and raw_key in semantic_cache
            scores = score_summary(raw_match)

            # Enrich scores from caches
            if was_scored and cached_sem:
                for field, cache_field in [
                    ("semantic_score", "semantic_score"),
                    ("library_similarity", "library_similarity"),
                    ("profile_similarity", "profile_similarity"),
                ]:
                    if scores.get(field, 0.0) == 0.0:
                        scores[field] = round(float(cached_sem.get(cache_field, 0.0)), 4)

            cached_llm_val = llm_cache.get(raw_key)
            if cached_llm_val is not None:
                scores["llm_score"] = float(cached_llm_val)

            sem_score = float(scores.get("semantic_score", 0.0))

            if not was_scored:
                drop_stage = "AGE_EXCLUDED"
                pub_utc = raw_match.get("published_utc") or raw_match.get("published", "")
                fetch_utc = raw_match.get("fetched_at", "")
                drop_reason = (
                    f"In raw cache but never scored — excluded by age/newness gate "
                    f"(--only-new / MAX_AGE_DAYS) before the semantic filter ran. "
                    f"Published: {pub_utc or 'unknown'}, Fetched: {fetch_utc or 'unknown'}."
                )
            elif sem_score < args.threshold:
                drop_stage = "FILTERED_OUT"
                drop_reason = (
                    f"Semantic score {sem_score:.4f} below filter threshold "
                    f"{args.threshold}. library_similarity="
                    f"{scores.get('library_similarity', 0.0):.4f}, "
                    f"profile_similarity={scores.get('profile_similarity', 0.0):.4f}"
                )
            else:
                # Passed filter — was it LLM-scored?
                drop_stage = "RANKED_OUT"
                llm_val = scores.get("llm_score")
                if llm_val is not None:
                    nov_data = novelty_cache.get(raw_key, {})
                    novelty = float(nov_data.get("novelty_score", 0.0)) if isinstance(nov_data, dict) else 0.0
                    final = reconstruct_final_score(float(llm_val), novelty)
                    drop_reason = (
                        f"Passed filter (semantic={sem_score:.4f}), "
                        f"LLM scored {float(llm_val):.3f}, "
                        f"final={final:.3f} — below MIN_SCORE {args.min_score:.2f}"
                    )
                    scores["score"] = round(final, 4)
                else:
                    drop_reason = (
                        f"Passed filter (semantic={sem_score:.4f}) "
                        f"but was never LLM-scored — likely fell outside "
                        f"--keep-top-n or --max-llm cutoff."
                    )
        else:
            drop_stage  = "NOT_IN_CACHE"
            drop_reason = "Article was not fetched at all — check feeds.txt or Crossref terms."
            scores      = {}

        false_negatives.append({
            "ground_truth": gt,
            "drop_stage":   drop_stage,
            "drop_reason":  drop_reason,
            "scores":       scores,
            "raw_title":    raw_match.get("title") if raw_match else None,
            "raw_feed":     raw_match.get("feed")  if raw_match else None,
        })

    # ── False positives: in ranked output but not on Bluesky ──────────────────
    bluesky_urls = {normalise_url(gt["url"]) for gt in ground_truth}
    bluesky_ids: set[str] = set()
    for gt in ground_truth:
        bluesky_ids |= extract_identifiers(gt["url"])

    false_positives = []
    for article in ranked:
        if float(article.get("score", 0.0) or 0.0) < args.min_score:
            continue
        a_url = normalise_url(article.get("link") or article.get("url", ""))
        a_ids = set()
        for field in ("link", "url", "normalized_link", "doi"):
            a_ids |= extract_identifiers(article.get(field, "") or "")
        if (a_url and a_url in bluesky_urls) or (a_ids and (a_ids & bluesky_ids)):
            continue
        false_positives.append({
            "title": article.get("title"),
            "link":  article.get("link"),
            **score_summary(article),
        })

    # ── Print summary ──────────────────────────────────────────────────────────
    total   = len(ground_truth)
    n_tp    = len(true_positives)
    n_fn    = len(false_negatives)
    n_fp    = len(false_positives)
    recall  = n_tp / total if total else 0.0
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0

    tp_current = sum(1 for tp in true_positives if tp["run"] == "current")
    tp_prior   = sum(1 for tp in true_positives if tp["run"] == "prior")

    print("=" * 65)
    print(f"EVALUATION RESULTS  (last {args.days_back} days, handle: @{args.handle})")
    print("=" * 65)
    print(f"  Ground truth articles (from Bluesky): {total}")
    print(f"  True positives  (pipeline caught):    {n_tp}")
    print(f"    ├─ In current run's output:         {tp_current}")
    print(f"    └─ Surfaced in a prior run:          {tp_prior}")
    print(f"  False negatives (pipeline missed):    {n_fn}")
    print(f"  False positives (pipeline extra):     {n_fp}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  Precision: {precision:.1%}  (vs. your Bluesky posts as ground truth)")
    print()

    # Drop-stage breakdown for false negatives
    stages: dict[str, int] = {}
    for fn in false_negatives:
        stages[fn["drop_stage"]] = stages.get(fn["drop_stage"], 0) + 1
    if stages:
        print("  False negative breakdown:")
        for stage, count in sorted(stages.items()):
            print(f"    {stage}: {count}")
        print()

    # Prior-run TPs (brief summary)
    if tp_prior:
        print("─" * 65)
        print(f"PRIOR-RUN TRUE POSITIVES ({tp_prior} articles surfaced in earlier digests):")
        print("─" * 65)
        for tp in true_positives:
            if tp["run"] != "prior":
                continue
            gt = tp["ground_truth"]
            pa = tp["pipeline_article"]
            print(f"  ✓ {(gt.get('title') or gt.get('url',''))[:90]}")
            print(f"    llm={pa.get('llm_score', 'n/a')}  "
                  f"final={pa.get('score', 'n/a')}  "
                  f"posted={gt['created_at'][:10]}")

    # Detailed false negatives
    if false_negatives:
        print()
        print("─" * 65)
        print("FALSE NEGATIVES (articles you posted that the pipeline missed):")
        print("─" * 65)
        for fn in false_negatives:
            gt = fn["ground_truth"]
            print(f"\n  [{fn['drop_stage']}]")
            print(f"  URL:    {gt['url']}")
            if gt.get("title"):
                print(f"  Title:  {gt['title'][:120]}")
            print(f"  Posted: {gt['created_at'][:10]}")
            print(f"  Reason: {fn['drop_reason']}")
            if fn.get("raw_feed"):
                print(f"  Feed:   {fn['raw_feed']}")
            if fn.get("scores"):
                s = fn["scores"]
                print(f"  Scores: semantic={s.get('semantic_score', '?'):.4f}  "
                      f"library={s.get('library_similarity', '?'):.4f}  "
                      f"profile={s.get('profile_similarity', '?'):.4f}  "
                      f"llm={s.get('llm_score') or 'n/a'}")

    # ── Save full report ───────────────────────────────────────────────────────
    report = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "handle":         args.handle,
        "days_back":      args.days_back,
        "min_score":      args.min_score,
        "total_gt":       total,
        "true_positives": n_tp,
        "tp_current_run": tp_current,
        "tp_prior_run":   tp_prior,
        "false_negatives":n_fn,
        "false_positives":n_fp,
        "recall":         round(recall, 4),
        "precision":      round(precision, 4),
        "drop_stage_counts": stages,
        "fn_details":     false_negatives,
        "tp_details":     true_positives,
        "fp_details":     false_positives[:50],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull report saved to {args.report_out}")

    # ── Tuning recommendations ─────────────────────────────────────────────────
    filtered_out = [fn for fn in false_negatives if fn["drop_stage"] == "FILTERED_OUT"]
    not_cached   = [fn for fn in false_negatives if fn["drop_stage"] == "NOT_IN_CACHE"]
    ranked_out   = [fn for fn in false_negatives if fn["drop_stage"] == "RANKED_OUT"]
    age_excluded = [fn for fn in false_negatives if fn["drop_stage"] == "AGE_EXCLUDED"]

    if age_excluded:
        print(f"\n⚠  AGE GATE: {len(age_excluded)} articles were in the raw cache but excluded by")
        print("   the --only-new / MAX_AGE_DAYS filter before scoring.")
        print("   → Consider increasing MAX_AGE_DAYS or running RANK_ONLY_NEW=0.")
        for fn in age_excluded[:5]:
            gt = fn["ground_truth"]
            print(f"   - {gt.get('title') or gt.get('url', '')[:80]}")
            print(f"     {fn['drop_reason']}")

    if filtered_out:
        scores_list = [fn["scores"].get("semantic_score", 0.0) for fn in filtered_out if fn.get("scores")]
        if scores_list:
            avg = sum(scores_list) / len(scores_list)
            max_score = max(scores_list)
            print(f"\n⚠  FILTER TUNING: {len(filtered_out)} articles filtered out.")
            print(f"   Their semantic scores: avg={avg:.4f}, max={max_score:.4f}")
            if max_score > 0.20:
                print(f"   → Consider lowering --threshold to {max_score - 0.02:.2f}")

    if not_cached:
        print(f"\n⚠  FETCH COVERAGE: {len(not_cached)} articles were never fetched.")
        print("   Check feeds.txt and Crossref terms.")
        for fn in not_cached[:5]:
            print(f"   - {fn['ground_truth'].get('url', '')}")

    if ranked_out:
        print(f"\n⚠  RANKER: {len(ranked_out)} articles passed the filter but scored below MIN_SCORE={args.min_score:.2f}.")
        for fn in ranked_out[:5]:
            gt = fn["ground_truth"]
            s = fn.get("scores", {})
            print(f"   - {(gt.get('title') or '')[:80]}")
            print(f"     llm={s.get('llm_score', 'n/a')}  final={s.get('score', 'n/a')}")


if __name__ == "__main__":
    main()
