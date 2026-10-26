#!/usr/bin/env bash
# run_pipeline.sh
# Runs the full literature curation pipeline.
#
# Deleted scripts (functionality merged):
#   supplement_from_crossref.py  →  fetch_feeds.py
#   supplement_high_volume.py    →  fetch_feeds.py
#   semantic_filter.py           →  filter_and_rank.py
#   embed_articles.py            →  removed (was orphaned, never called)
#   rank_articles.py             →  filter_and_rank.py
set -euo pipefail

BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$BIN_DIR/.." && pwd)"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ── Tunable knobs (override via env) ──────────────────────────────────────────
MAX_LLM="${MAX_LLM:-2000}"
RANK_ONLY_NEW="${RANK_ONLY_NEW:-1}"
export MAX_AGE_DAYS="${MAX_AGE_DAYS:-3}"
VALIDATE_FEEDS="${VALIDATE_FEEDS:-0}"
SEMANTIC_THRESHOLD="${SEMANTIC_THRESHOLD:-0.25}"
MAX_FILTERED="${MAX_FILTERED:-2000}"
MIN_SCORE="${MIN_SCORE:-0.60}"
FETCH_WORKERS="${FETCH_WORKERS:-4}"
FETCH_TIMEOUT="${FETCH_TIMEOUT:-45}"
RANK_KEEP_TOP_N="${RANK_KEEP_TOP_N:-2000}"
NOVEL_THRESHOLD="${NOVEL_THRESHOLD:-0.60}"

# Digest output tiers (score-based, replaces hard count cutoffs)
export TOP_PICKS_SCORE="${TOP_PICKS_SCORE:-0.75}"
export MIN_SCORE

# Crossref knobs (now part of fetch_feeds.py)
CROSSREF_DAYS_BACK="${CROSSREF_DAYS_BACK:-21}"
CROSSREF_ROWS="${CROSSREF_ROWS:-100}"
CROSSREF_MAX_PER_TERM="${CROSSREF_MAX_PER_TERM:-150}"
CROSSREF_MAILTO="${CROSSREF_MAILTO:-}"
SKIP_CROSSREF="${SKIP_CROSSREF:-0}"
CROSSREF_TERMS="${CROSSREF_TERMS:-config/crossref_terms.txt}"

# High-volume source knobs (bioRxiv + Nature, now part of fetch_feeds.py)
SKIP_BIORXIV="${SKIP_BIORXIV:-0}"
SKIP_NATURE="${SKIP_NATURE:-0}"
SKIP_ASM="${SKIP_ASM:-0}"
BIORXIV_TERMS="${BIORXIV_TERMS:-config/biorxiv_terms.txt}"
NATURE_JOURNALS="${NATURE_JOURNALS:-config/crossref_issn_fallbacks.txt}"

cd "$PROJECT_ROOT"
mkdir -p data/outputs/markdown data/outputs/html data/outputs/csv data/outputs/reports

# ── Read previous run timestamp ────────────────────────────────────────────────
PREV_RUN_UTC="$($PYTHON_BIN - <<'PY'
from pathlib import Path
import json
p = Path('data/cache/fetch_state.json')
if p.exists():
    try:
        obj = json.loads(p.read_text())
        print(obj.get('last_run_utc',''))
    except Exception:
        print('')
else:
    print('')
PY
)"
RUN_STARTED_UTC="$($PYTHON_BIN - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
PY
)"

# ── Optional: validate feeds ───────────────────────────────────────────────────
if [[ "$VALIDATE_FEEDS" == "1" && -f "$SCRIPTS_DIR/validate_feeds.py" ]]; then
  echo "=== Validating feeds ==="
  "$PYTHON_BIN" "$SCRIPTS_DIR/validate_feeds.py" \
    --feeds config/feeds.txt \
    --max-workers "$FETCH_WORKERS" \
    --timeout-sec "$FETCH_TIMEOUT"
fi

# ── Prune old cache entries ────────────────────────────────────────────────────
if [[ -f "$SCRIPTS_DIR/prune_cache.py" ]]; then
  echo "=== Pruning cache ==="
  "$PYTHON_BIN" "$SCRIPTS_DIR/prune_cache.py" --max-age-days "$MAX_AGE_DAYS"
fi

# ── Fetch: RSS + Crossref in one pass ─────────────────────────────────────────
echo "=== Fetching feeds + Crossref + bioRxiv + Nature ==="
FETCH_ARGS=(
  --max-workers "$FETCH_WORKERS"
  --timeout-sec "$FETCH_TIMEOUT"
  --last-run-utc "$PREV_RUN_UTC"
  --run-started-utc "$RUN_STARTED_UTC"
  --crossref-days-back "$CROSSREF_DAYS_BACK"
  --crossref-rows "$CROSSREF_ROWS"
  --crossref-max-per-term "$CROSSREF_MAX_PER_TERM"
)
if [[ -n "$CROSSREF_MAILTO" ]]; then
  FETCH_ARGS+=(--crossref-mailto "$CROSSREF_MAILTO")
fi
if [[ "$SKIP_CROSSREF" == "1" ]]; then
  FETCH_ARGS+=(--skip-crossref)
fi
if [[ -f "$CROSSREF_TERMS" ]]; then
  FETCH_ARGS+=(--crossref-terms "$CROSSREF_TERMS")
fi
if [[ "$SKIP_BIORXIV" == "1" ]]; then
  FETCH_ARGS+=(--skip-biorxiv)
fi
if [[ "$SKIP_NATURE" == "1" ]]; then
  FETCH_ARGS+=(--skip-nature)
fi
if [[ "$SKIP_ASM" == "1" ]]; then
  FETCH_ARGS+=(--skip-asm)
fi
if [[ -f "$BIORXIV_TERMS" ]]; then
  FETCH_ARGS+=(--biorxiv-terms "$BIORXIV_TERMS")
fi
if [[ -f "$NATURE_JOURNALS" ]]; then
  FETCH_ARGS+=(--nature-journals "$NATURE_JOURNALS")
fi
"$PYTHON_BIN" "$SCRIPTS_DIR/fetch_feeds.py" "${FETCH_ARGS[@]}"

# ── Filter + novelty score + rank in one pass ──────────────────────────────────
echo "=== Filtering and ranking ==="
FILTER_RANK_ARGS=(
  --threshold "$SEMANTIC_THRESHOLD"
  --max-filtered "$MAX_FILTERED"
  --max-llm "$MAX_LLM"
  --keep-top-n "$RANK_KEEP_TOP_N"
  --min-score "$MIN_SCORE"
  --novel-threshold "$NOVEL_THRESHOLD"
)
if [[ "$RANK_ONLY_NEW" == "1" ]]; then
  FILTER_RANK_ARGS+=(--only-new --fallback-to-all)
fi
TOKENIZERS_PARALLELISM=false "$PYTHON_BIN" "$SCRIPTS_DIR/filter_and_rank.py" "${FILTER_RANK_ARGS[@]}"

# ── Generate digest ────────────────────────────────────────────────────────────
echo "=== Generating digest ==="
"$PYTHON_BIN" "$SCRIPTS_DIR/generate_markdown.py"

echo ""
echo "=== Done. Outputs in: ==="
echo "  $PROJECT_ROOT/data/outputs/markdown"
echo "  $PROJECT_ROOT/data/outputs/html"
echo "  $PROJECT_ROOT/data/outputs/csv"
