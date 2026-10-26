#!/usr/bin/env python3
"""
Filter RSS articles semantically then rank with heuristics + Claude Haiku.

Replaces: semantic_filter.py, embed_articles.py (orphaned), rank_articles.py

Embeddings: sentence-transformers all-mpnet-base-v2 (local, no API cost).
Ranking:    Claude Haiku via Anthropic SDK (with prompt caching for cost reduction).
No OpenAI dependency.
"""
from __future__ import annotations

# Must be set before any torch/transformers imports to prevent segfault
# caused by faiss+PyTorch BLAS conflict on Apple Silicon
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load .env before any API clients are instantiated
from pathlib import Path as _Path
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(_Path(__file__).resolve().parents[1] / ".env")

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import subprocess
import sys
import tempfile

import anthropic
import numpy as np

from common_paths import (
    FILTERED_JSON,
    INTEREST_PROFILE_TXT,
    LIBRARY_FAISS_INDEX,
    LLM_SCORES_JSON,
    NEW_PUBLISHED_IDS_JSON,
    NOVELTY_SCORES_JSON,
    PROFILE_EMBEDDING_JSON,
    RAW_FEEDS_JSON,
    RANKED_JSON,
    RANKING_PROMPT_TXT,
    SEMANTIC_FILTER_STATE_JSON,
    SEMANTIC_SCORES_JSON,
    ensure_project_dirs,
    load_json,
    load_text,
    save_json,
)

# ── Embedding / FAISS constants ────────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBED_BATCH_SIZE = 8    # small batches to avoid PyTorch/faiss BLAS conflict on Apple Silicon

# ── Filter constants ───────────────────────────────────────────────────────────
# Profile weight raised vs original (0.35→0.50): we care more about topic match
# than similarity to existing library, which would suppress novel discoveries.
PROFILE_WEIGHT = 0.50
LIBRARY_WEIGHT = 0.50
# Lowered from 0.42 → 0.30.  The LLM ranker handles precision; the filter's job
# is recall.  Papers in the 0.30–0.42 band were the most common false negatives.
SIMILARITY_THRESHOLD = 0.30
MAX_FILTERED = 400
TOP_K = 5

# ── Ranking constants ──────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-haiku-4-5"
LLM_MAX_RETRIES = 8
DEFAULT_MAX_LLM = 120
DEFAULT_KEEP_TOP_N = 150
DEFAULT_MIN_SCORE = 0.30

NEGATIVE_TERMS = {"erratum", "correction", "retraction", "editorial", "book review", "table of contents"}
DEFAULT_POSITIVE_TERMS = {
    "metagenom", "microbi", "archaea", "biogeochem", "genome",
    "metabolic", "ocean", "marine", "ecology", "transcriptom",
}
GENERIC_TOKENS = {
    "models", "model", "systems", "system", "biology", "agent",
    "based", "scale", "cycles", "cycle", "mechanistic", "analysis",
    "approach", "approaches", "framework", "frameworks", "dynamics",
    "function", "functions", "functional", "process", "processes",
    "community", "communities", "microbial", "microbes", "microbe",
}

anthropic_client = anthropic.Anthropic()

# Path to the standalone embedder script (same directory as this script)
_EMBEDDER_SCRIPT = Path(__file__).parent / "embed_texts.py"


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Semantic filter + rank pipeline (replaces semantic_filter + rank_articles)"
    )
    # inputs
    p.add_argument("--raw", type=Path, default=RAW_FEEDS_JSON)
    p.add_argument("--interest-profile", type=Path, default=INTEREST_PROFILE_TXT)
    p.add_argument("--ranking-prompt", type=Path, default=RANKING_PROMPT_TXT)
    p.add_argument("--library-index", type=Path, default=LIBRARY_FAISS_INDEX)
    p.add_argument("--new-published-ids", type=Path, default=NEW_PUBLISHED_IDS_JSON)
    # outputs / caches
    p.add_argument("--filtered", type=Path, default=FILTERED_JSON)
    p.add_argument("--ranked", type=Path, default=RANKED_JSON)
    p.add_argument("--semantic-cache", type=Path, default=SEMANTIC_SCORES_JSON)
    p.add_argument("--novelty-cache", type=Path, default=NOVELTY_SCORES_JSON)
    p.add_argument("--llm-cache", type=Path, default=LLM_SCORES_JSON)
    p.add_argument("--profile-cache", type=Path, default=PROFILE_EMBEDDING_JSON)
    p.add_argument("--state-file", type=Path, default=SEMANTIC_FILTER_STATE_JSON)
    # filter tuning
    p.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    p.add_argument("--max-filtered", type=int, default=MAX_FILTERED)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE)
    p.add_argument("--profile-weight", type=float, default=PROFILE_WEIGHT)
    # rank tuning
    p.add_argument("--max-llm", type=int, default=DEFAULT_MAX_LLM)
    p.add_argument("--keep-top-n", type=int, default=DEFAULT_KEEP_TOP_N)
    p.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    p.add_argument("--novel-threshold", type=float, default=0.60)
    # mode flags
    p.add_argument("--only-new", action="store_true")
    p.add_argument("--fallback-to-all", action="store_true")
    return p.parse_args()


# ── Embedding helpers ──────────────────────────────────────────────────────────

def embed_texts_subprocess(texts: list[str]) -> np.ndarray:
    """
    Embed texts by calling embed_texts.py in a subprocess.
    This keeps torch and faiss in separate processes, avoiding the
    BLAS/MKL conflict that causes segfaults on Apple Silicon.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        texts_path = Path(tmpdir) / "texts.json"
        vecs_path  = Path(tmpdir) / "vecs.npy"
        texts_path.write_text(json.dumps(texts), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(_EMBEDDER_SCRIPT),
             "--texts", str(texts_path),
             "--out",   str(vecs_path)],
            stderr=None,   # let stderr print directly so progress is visible
            stdout=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Embedder subprocess failed (exit {result.returncode})")
        return np.load(str(vecs_path))


def load_or_create_profile_vector(profile_path: Path, cache_path: Path) -> np.ndarray:
    cached = load_json(cache_path, {})
    if cached.get("model") == EMBED_MODEL and isinstance(cached.get("embedding"), list):
        return np.array(cached["embedding"], dtype="float32")
    vec = embed_texts_subprocess([profile_path.read_text().strip()])[0]
    save_json(cache_path, {"model": EMBED_MODEL, "embedding": vec.tolist()})
    return vec


def embed_needed_articles(
    articles: list[dict[str, Any]],
    semantic_cache: dict[str, Any],
    novelty_cache: dict[str, Any],
) -> dict[str, np.ndarray]:
    """
    Embed all articles that lack EITHER a semantic or novelty cached score
    in a single subprocess call.  Returns {article_key: vector}.
    """
    needed_keys: dict[str, str] = {}  # key → text
    for a in articles:
        key = article_key(a)
        if not key:
            continue
        if key in semantic_cache and key in novelty_cache:
            continue
        text = article_text_for_embedding(a)
        if text.strip():
            needed_keys[key] = text

    if not needed_keys:
        return {}

    keys_list = list(needed_keys.keys())
    texts_list = [needed_keys[k] for k in keys_list]
    print(f"  Embedding {len(texts_list)} articles in one pass...")
    vecs = embed_texts_subprocess(texts_list)
    return {k: vecs[i] for i, k in enumerate(keys_list)}


# ── Article key / text helpers ─────────────────────────────────────────────────

def article_key(article: dict[str, Any]) -> str:
    return article.get("content_hash") or article.get("id") or ""


def article_text_for_embedding(a: dict[str, Any]) -> str:
    return (
        f"Title: {(a.get('title') or '').strip()}\n"
        f"Abstract: {(a.get('content') or '').strip()}"
    ).strip()


# ── Semantic filtering ─────────────────────────────────────────────────────────

def run_semantic_filter(
    articles: list[dict[str, Any]],
    args: argparse.Namespace,
    index: "faiss.Index",
    profile_vec: np.ndarray,
    score_cache: dict[str, Any],
    precomputed_vecs: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    profile_vec_2d = np.array([profile_vec], dtype="float32")
    library_weight = 1.0 - args.profile_weight

    # Deduplicate by normalised title before scoring (same article, different hashes)
    seen_titles: set[str] = set()
    deduped_articles: list[dict[str, Any]] = []
    for a in articles:
        t = re.sub(r'\s+', ' ', (a.get('title') or '').strip().lower())
        if t and t in seen_titles:
            continue
        if t:
            seen_titles.add(t)
        deduped_articles.append(a)
    if len(deduped_articles) < len(articles):
        print(f"  Deduped {len(articles) - len(deduped_articles)} duplicate titles.")
        articles = deduped_articles

    uncached = [a for a in articles if article_key(a) not in score_cache]
    if uncached:
        # Use pre-computed vectors (no second subprocess spawn)
        uncached_with_vecs = [(a, precomputed_vecs[article_key(a)])
                              for a in uncached if article_key(a) in precomputed_vecs]
        if uncached_with_vecs:
            keys_arr = [article_key(a) for a, _ in uncached_with_vecs]
            vecs_arr = np.array([v for _, v in uncached_with_vecs], dtype="float32")

            for i in range(0, len(uncached_with_vecs), args.batch_size):
                batch = uncached_with_vecs[i : i + args.batch_size]
                vecs  = vecs_arr[i : i + args.batch_size]
                sims, _ = index.search(vecs, args.top_k)
                profile_scores = (vecs @ profile_vec_2d.T).reshape(-1)
                for (article, _vec), lib_row, profile_score in zip(batch, sims, profile_scores):
                    library_similarity = float(np.max(lib_row)) if len(lib_row) else 0.0
                    combined = library_weight * library_similarity + args.profile_weight * float(profile_score)
                    score_cache[article_key(article)] = {
                        "library_similarity": library_similarity,
                        "profile_similarity": float(profile_score),
                        "semantic_score": combined,
                    }
            save_json(args.semantic_cache, score_cache)

    selected = []
    for a in articles:
        stats = score_cache.get(article_key(a), {})
        a["library_similarity"] = float(stats.get("library_similarity", 0.0))
        a["profile_similarity"] = float(stats.get("profile_similarity", 0.0))
        a["semantic_score"] = float(stats.get("semantic_score", 0.0))
        if a["semantic_score"] >= args.threshold:
            selected.append(a)

    selected.sort(key=lambda x: x.get("semantic_score", 0.0), reverse=True)
    return selected[: args.max_filtered]


# ── Novelty scoring (post-filter, pre-rank) ────────────────────────────────────

def run_novelty_scoring(
    articles: list[dict[str, Any]],
    index: "faiss.Index",
    novelty_cache: dict[str, Any],
    precomputed_vecs: dict[str, np.ndarray],
    top_k: int,
    novel_threshold: float,
    batch_size: int,
) -> None:
    """Adds novelty_score in-place. Uses pre-computed vectors from shared embedding pass."""
    to_score = [
        (i, article_key(a))
        for i, a in enumerate(articles)
        if article_key(a) not in novelty_cache
        and article_key(a) in precomputed_vecs
    ]

    if to_score:
        indices_list = [x[0] for x in to_score]
        keys = [x[1] for x in to_score]
        vecs = np.array([precomputed_vecs[k] for k in keys], dtype="float32")

        for batch_start in range(0, len(to_score), batch_size):
            batch_indices = indices_list[batch_start : batch_start + batch_size]
            batch_keys = keys[batch_start : batch_start + batch_size]
            batch_vecs = vecs[batch_start : batch_start + batch_size]
            D, _I = index.search(batch_vecs, max(1, top_k))

            for row_idx, article_idx in enumerate(batch_indices):
                sims = [float(x) for x in D[row_idx].tolist() if not math.isnan(float(x))]
                avg_topk = float(sum(sims) / len(sims)) if sims else 0.0
                novelty = round(max(0.0, min(1.0, 1.0 - avg_topk)), 4)
                payload = {
                    "max_library_similarity": round(max(sims) if sims else 0.0, 4),
                    "avg_topk_similarity": round(avg_topk, 4),
                    "novelty_score": novelty,
                    "is_novel_candidate": novelty >= novel_threshold,
                }
                articles[article_idx].update(payload)
                novelty_cache[batch_keys[row_idx]] = payload

    # Apply cached values for articles that were already scored
    for article in articles:
        key = article_key(article)
        if key in novelty_cache and "novelty_score" not in article:
            article.update(novelty_cache[key])


# ── Lexical scoring ────────────────────────────────────────────────────────────

def load_interest_terms(path: Path) -> set[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return set(DEFAULT_POSITIVE_TERMS)

    terms: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        low = line.lower()
        if len(low) >= 4:
            terms.add(low)
        for token in re.findall(r"[a-z0-9][a-z0-9\-_/+]*", low):
            token = token.strip("-_/+")
            if len(token) < 5 or token in GENERIC_TOKENS:
                continue
            terms.add(token)

    return terms or set(DEFAULT_POSITIVE_TERMS)


def cheap_lexical_score(article: dict[str, Any], positive_terms: set[str]) -> float:
    text = f"{article.get('title', '')} {article.get('content', '')}".lower()

    # Normalise per-hit weight by vocabulary size so score doesn't saturate
    # for large interest profiles. Cap contribution at 0.40.
    per_hit = min(0.08, 0.40 / max(len(positive_terms), 1))
    score = 0.0
    for term in positive_terms:
        if term in text:
            score += per_hit
    for term in NEGATIVE_TERMS:
        if term in text:
            score -= 0.35
    if article.get("published") or article.get("published_utc"):
        score += 0.03
    if article.get("feed"):
        score += 0.02
    return max(-1.0, min(1.0, score))


# ── LLM ranking (Claude Haiku — batched, prompt-cached) ────────────────────────
# Prompt caching: the ranking prompt is sent as a system message with
# cache_control=ephemeral so it's billed once per 5-minute window.
# Subsequent calls read from cache at 10% of the standard input cost.
# With batch size 20, a full 2000-article run makes ~100 API calls —
# well within the 5-minute cache TTL.

LLM_BATCH_SIZE = 20  # articles per API call — larger batches amortise per-call overhead


def _format_article_for_llm(article: dict[str, Any], idx: int) -> str:
    content = (article.get("content") or "")[:3000]
    return (
        f"--- Article {idx} ---\n"
        f"Title: {article.get('title', '')}\n"
        f"Published: {article.get('published_utc') or article.get('published', '')}\n"
        f"Feed: {article.get('feed', '')}\n"
        f"Abstract: {content}\n"
    )


def _parse_batch_scores(text: str, n: int) -> list[float | None]:
    """Parse JSON array of scores from batch response. Returns list of n scores."""
    # Try JSON array first: [{"score": 0.85}, {"score": 0.72}, ...]
    try:
        arr = json.loads(text)
        if isinstance(arr, list) and len(arr) == n:
            return [max(0.0, min(1.0, float(item.get("score", item) if isinstance(item, dict) else item)))
                    for item in arr]
    except Exception:
        pass
    # Try JSON object with "scores" key: {"scores": [0.85, 0.72, ...]}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "scores" in obj:
            scores = obj["scores"]
            if isinstance(scores, list) and len(scores) == n:
                return [max(0.0, min(1.0, float(s.get("score", s) if isinstance(s, dict) else s)))
                        for s in scores]
    except Exception:
        pass
    # Fall back to extracting all float-like numbers
    matches = re.findall(r"([01](?:\.\d+)?)", text)
    if len(matches) >= n:
        return [max(0.0, min(1.0, float(m))) for m in matches[:n]]
    return [None] * n  # parse failed — caller will retry individually


def llm_score_batch(articles: list[dict[str, Any]], prompt_text: str) -> list[float]:
    """Score a batch of articles in one API call. Falls back to individual scoring on failure.

    Uses prompt caching: the ranking prompt is sent as a system message with
    cache_control, so it's only processed (and billed) once per 5-minute window.
    Subsequent calls within the window read from cache at 10% of the input cost.
    """
    n = len(articles)
    article_blocks = "\n".join(
        _format_article_for_llm(a, i + 1) for i, a in enumerate(articles)
    )
    user_message = (
        f"Score each of the following {n} articles. "
        f"Return a JSON array of {n} objects, one per article in order: "
        f'[{{"score": 0.XX}}, ...]\n\n'
        f"{article_blocks}"
    )
    delay = 1.0
    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=32 * n,
                system=[{
                    "type": "text",
                    "text": prompt_text,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_message}],
            )
            text = response.content[0].text.strip() if response.content else ""
            scores = _parse_batch_scores(text, n)
            if all(s is not None for s in scores):
                return scores  # type: ignore[return-value]
            # Partial parse — fill in missing with individual calls
            result = []
            for i, s in enumerate(scores):
                if s is not None:
                    result.append(s)
                else:
                    result.append(_llm_score_single(articles[i], prompt_text))
            return result
        except anthropic.RateLimitError:
            if attempt == LLM_MAX_RETRIES - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
        except anthropic.APIError:
            if attempt == LLM_MAX_RETRIES - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    return [0.5] * n


def _llm_score_single(article: dict[str, Any], prompt_text: str) -> float:
    """Fallback: score a single article when batch parsing fails."""
    content = (article.get("content") or "")[:3500]
    user_message = (
        f"Title: {article.get('title', '')}\n"
        f"Published: {article.get('published_utc') or article.get('published', '')}\n"
        f"Feed: {article.get('feed', '')}\n"
        f"Abstract: {content}\n"
    )
    delay = 1.0
    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=32,
                system=[{
                    "type": "text",
                    "text": prompt_text,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_message}],
            )
            text = response.content[0].text.strip() if response.content else ""
            try:
                return max(0.0, min(1.0, float(json.loads(text)["score"])))
            except Exception:
                pass
            m = re.search(r"([01](?:\.\d+)?)", text)
            return max(0.0, min(1.0, float(m.group(1)))) if m else 0.5
        except (anthropic.RateLimitError, anthropic.APIError):
            if attempt == LLM_MAX_RETRIES - 1:
                return 0.5
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    return 0.5


# ── Ranking ────────────────────────────────────────────────────────────────────

def run_ranking(
    articles: list[dict[str, Any]],
    args: argparse.Namespace,
    positive_terms: set[str],
    prompt_text: str,
    score_cache: dict[str, Any],
) -> list[dict[str, Any]]:
    # Pre-LLM composite score
    for article in articles:
        article["lexical_score"] = cheap_lexical_score(article, positive_terms)
        novelty = float(article.get("novelty_score", 0.0) or 0.0)
        article["pre_llm_score"] = (
            0.60 * float(article.get("semantic_score", 0.0))
            + 0.25 * float(article.get("profile_similarity", 0.0))
            + 0.05 * float(article.get("lexical_score", 0.0))
            + 0.10 * novelty
        )

    articles.sort(key=lambda x: x.get("pre_llm_score", 0.0), reverse=True)
    llm_candidates = articles[: args.max_llm]

    # Separate cached from uncached
    uncached_batch: list[tuple[int, dict[str, Any]]] = []
    for idx, article in enumerate(llm_candidates):
        key = article_key(article)
        cached = score_cache.get(key)
        if cached is not None:
            article["llm_score"] = float(cached)
            article["scored_by_llm"] = True
        else:
            uncached_batch.append((idx, article))

    # Score uncached articles in batches of LLM_BATCH_SIZE
    total_uncached = len(uncached_batch)
    if total_uncached:
        print(f"  Scoring {total_uncached} uncached articles in batches of {LLM_BATCH_SIZE}...")
    for batch_start in range(0, total_uncached, LLM_BATCH_SIZE):
        batch = uncached_batch[batch_start : batch_start + LLM_BATCH_SIZE]
        batch_articles = [a for _, a in batch]
        batch_num = batch_start // LLM_BATCH_SIZE + 1
        total_batches = (total_uncached + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE
        print(f"  LLM batch {batch_num}/{total_batches} ({len(batch)} articles)")
        scores = llm_score_batch(batch_articles, prompt_text)
        for (orig_idx, article), score in zip(batch, scores):
            key = article_key(article)
            score_cache[key] = score
            article["llm_score"] = score
            article["scored_by_llm"] = True
        save_json(args.llm_cache, score_cache)

    for article in articles[args.max_llm :]:
        article["llm_score"] = None
        article["scored_by_llm"] = False

    for article in articles:
        llm_component = (
            float(article["llm_score"])
            if article.get("llm_score") is not None
            else float(article.get("pre_llm_score", 0.0))
        )
        novelty = float(article.get("novelty_score", 0.0) or 0.0)
        article["score"] = (
            0.90 * llm_component
            + 0.10 * novelty
        )
        article["discovery_score"] = round(
            0.75 * float(article.get("llm_score") or article.get("pre_llm_score", 0.0))
            + 0.25 * novelty,
            4,
        )

    articles.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    llm_scored   = [a for a in articles if a.get("scored_by_llm")]
    unllm_scored = [a for a in articles if not a.get("scored_by_llm")]

    kept_llm  = [a for a in llm_scored if float(a.get("score", 0.0)) >= args.min_score]
    kept_rest = unllm_scored[: max(0, args.keep_top_n - len(kept_llm))]
    kept = (kept_llm + kept_rest)[: args.keep_top_n]

    if not kept:
        kept = articles[: min(args.keep_top_n, 25)]
        print(f"  No articles met min-score {args.min_score:.2f}; falling back to top {len(kept)}.")

    return kept


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_project_dirs()
    args = parse_args()

    if not args.library_index.exists():
        raise FileNotFoundError(f"Missing library FAISS index: {args.library_index}")

    # ── Load raw articles ──────────────────────────────────────────────────────
    all_articles = load_json(args.raw, [])
    if not all_articles:
        save_json(args.filtered, [])
        save_json(args.ranked, [])
        print("No raw articles to process.")
        return

    used_fallback = False
    if args.only_new:
        # Use publication date as the primary "newness" criterion.
        # This catches articles that entered the cache in previous runs but
        # were published recently — e.g. Crossref supplements fetched overnight.
        # Fall back to fetch timestamp for articles with no publication date.
        state = load_json(args.state_file, {})
        prev_run_utc = load_json(args.state_file, {}).get("last_run_utc", "")
        fetch_state = load_json(
            args.state_file.parent / "fetch_state.json", {}
        )
        prev_run_utc = fetch_state.get("previous_run_utc") or fetch_state.get("last_run_utc") or ""

        # Parse previous run timestamp
        prev_run_ts: float | None = None
        if prev_run_utc:
            import re as _re
            m = _re.match(r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})Z", prev_run_utc)
            if m:
                from datetime import datetime as _dt, timezone as _tz
                prev_run_ts = _dt.strptime(prev_run_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_tz.utc).timestamp()

        # Also keep articles from new_published_ids (fetched this run)
        new_ids = set(load_json(args.new_published_ids, []))

        # Max age for "recent" articles.
        # Use a short window (3 days) on daily runs to avoid reprocessing the
        # entire 30-day cache. The supplement script handles backfill separately.
        # Override via MAX_AGE_DAYS env var if needed (e.g. after a long gap).
        import os as _os
        max_age_days = int(_os.environ.get("MAX_AGE_DAYS", "3"))
        min_pub_ts = time.time() - (max_age_days * 86400)

        def is_recent(a: dict) -> bool:
            pub_ts = a.get("published_ts")
            fetch_ts = a.get("timestamp")
            # Reject articles with a known publication date that's too old,
            # even if they were freshly fetched this run.
            if pub_ts is not None:
                try:
                    if float(pub_ts) < min_pub_ts:
                        return False
                except Exception:
                    pass
            # Include if fetched this run (and not excluded above)
            if a.get("id") in new_ids:
                return True
            # Include if published within max_age_days
            if pub_ts is not None:
                try:
                    return float(pub_ts) >= min_pub_ts
                except Exception:
                    pass
            # No pub date — use fetch timestamp if available
            if fetch_ts is not None and prev_run_ts is not None:
                try:
                    return float(fetch_ts) > prev_run_ts
                except Exception:
                    pass
            return False

        articles = [a for a in all_articles if is_recent(a)]

        if articles:
            print(f"Processing recently published articles: {len(articles)} candidates "
                  f"(published within {max_age_days} days).")
        elif args.fallback_to_all:
            articles = all_articles
            used_fallback = True
            print("No recent articles found; falling back to full raw cache.")
        else:
            save_json(args.filtered, [])
            save_json(args.ranked, [])
            save_json(
                args.state_file,
                {
                    "last_run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "input_count": 0,
                    "output_count": 0,
                    "threshold": args.threshold,
                    "used_only_new": True,
                    "used_fallback": False,
                },
            )
            print("No recently published articles to process.")
            return
    else:
        articles = all_articles

    # ── Shared resources ───────────────────────────────────────────────────────
    # Import faiss after torch model loads to avoid BLAS conflict on Apple Silicon
    import faiss
    index = faiss.read_index(str(args.library_index))
    profile_vec = load_or_create_profile_vector(args.interest_profile, args.profile_cache)
    score_cache = load_json(args.semantic_cache, {})
    novelty_cache = load_json(args.novelty_cache, {})
    llm_cache = load_json(args.llm_cache, {})
    positive_terms = load_interest_terms(args.interest_profile)
    prompt_text = (
        load_text(args.ranking_prompt).strip()
        or "Score this paper for relevance to the user's research interests."
    )

    print(f"Loaded {len(positive_terms)} lexical interest terms.")

    # ── Shared embedding pass ──────────────────────────────────────────────────
    print(f"\n[0/3] Embedding articles that need scoring...")
    precomputed_vecs = embed_needed_articles(articles, score_cache, novelty_cache)
    if precomputed_vecs:
        print(f"  Embedded {len(precomputed_vecs)} articles (single subprocess call).")
    else:
        print("  All articles already cached — no embedding needed.")

    # ── Stage 1: Semantic filter ───────────────────────────────────────────────
    print(f"\n[1/3] Semantic filtering {len(articles)} articles (threshold={args.threshold})...")
    selected = run_semantic_filter(articles, args, index, profile_vec, score_cache, precomputed_vecs)
    save_json(args.filtered, selected)
    save_json(
        args.state_file,
        {
            "last_run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_count": len(articles),
            "output_count": len(selected),
            "threshold": args.threshold,
            "used_only_new": bool(args.only_new),
            "used_fallback": used_fallback,
        },
    )
    print(f"  Kept {len(selected)} of {len(articles)} articles after semantic filter.")

    if not selected:
        save_json(args.ranked, [])
        print("No articles passed the filter; nothing to rank.")
        return

    # ── Stage 2: Novelty scoring ───────────────────────────────────────────────
    print(f"\n[2/3] Scoring novelty for {len(selected)} articles...")
    run_novelty_scoring(
        selected, index, novelty_cache, precomputed_vecs,
        top_k=args.top_k,
        novel_threshold=args.novel_threshold,
        batch_size=args.batch_size,
    )
    save_json(args.novelty_cache, novelty_cache)
    novel_count = sum(1 for a in selected if a.get("is_novel_candidate"))
    print(f"  {novel_count} articles flagged as novel candidates.")

    # ── Stage 3: LLM ranking (batched) ─────────────────────────────────────────
    print(f"\n[3/3] Ranking (batched LLM scoring top {args.max_llm} via Claude Haiku, "
          f"batch_size={LLM_BATCH_SIZE})...")
    kept = run_ranking(selected, args, positive_terms, prompt_text, llm_cache)
    for article in kept:
        article["used_fallback"] = used_fallback

    save_json(args.ranked, kept)
    print(
        f"\nDone. Filtered: {len(selected)} → Ranked & kept: {len(kept)} "
        f"(min_score={args.min_score})."
    )
    if used_fallback:
        print("Note: digest based on cached articles (no new publications this run).")


if __name__ == "__main__":
    main()
