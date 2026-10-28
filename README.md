# RSS Agent Claude

An automated scientific literature curation pipeline that fetches papers from RSS feeds, Crossref, bioRxiv, Nature journals, and ASM Press, filters them semantically using embeddings, ranks them with an LLM (Claude), and produces a daily digest of the most relevant papers for your research interests.

## Origin and Attribution

This project is a substantially extended fork of [**rss_agent**](https://github.com/jrcasey/rss_agent) by **John Casey** (MIT License), which provided the original concept of an RSS-based literature scanner with LLM scoring.

**Development chain:**
1. **John Casey** — original `rss_agent`: RSS feed reader with LLM scoring, GitHub Actions, weekly email digest
2. **ChatGPT** — initial modifications and feature additions
3. **Claude** (Anthropic) — major refactoring, multi-source fetching, semantic filtering, FAISS novelty scoring, batched LLM ranking with prompt caching, and the current architecture

## How It Works

The pipeline runs in three stages, orchestrated by `run_pipeline.sh`:

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────┐
│  1. FETCH        │ ──▸ │  2. FILTER & RANK     │ ──▸ │  3. GENERATE DIGEST│
│  fetch_feeds.py  │     │  filter_and_rank.py   │     │  generate_markdown │
└─────────────────┘     └──────────────────────┘     └────────────────────┘
```

### Stage 1: Fetch (`fetch_feeds.py`)
- Parses RSS/Atom feeds from your `feeds.txt` list
- Supplements with Crossref API queries using your search terms
- Fetches bioRxiv preprints via Crossref DOI prefix queries, filtered by your keywords
- Fetches from Nature-family and other journals via Crossref ISSN queries
- Fetches from ASM Press journals via Crossref ISSN queries
- Deduplicates across all sources using DOI extraction
- Tracks fetch state to avoid re-processing

### Stage 2: Filter & Rank (`filter_and_rank.py`)
- **Semantic filtering**: Embeds article titles+abstracts using `sentence-transformers/all-mpnet-base-v2`, compares against your interest profile vector
- **Novelty scoring**: Uses a FAISS index of your personal paper library to score how novel each article is relative to what you've already read
- **LLM ranking**: Sends pre-filtered articles to Claude Haiku in batches of 20, with a detailed scoring prompt you customize. Uses prompt caching to minimize API costs
- **Combined scoring**: Final score = 0.90 × LLM score + 0.10 × novelty score

### Stage 3: Digest (`generate_markdown.py`)
- Sorts articles by score into tiers:
  - **Top Picks** (score ≥ 0.75): Papers you'd want to read in full
  - **Worth a Look** (score ≥ MIN_SCORE): Papers worth scanning
- Outputs Markdown, HTML, and CSV formats
- Tracks previously shown articles to avoid repeats (365-day window)

## Requirements

- **Python 3.10+**
- **Anthropic API key** (for Claude Haiku ranking) — [Get one here](https://console.anthropic.com/)
- ~2 GB disk for the embedding model (downloaded on first run)
- ~500 MB RAM for embedding operations

### Platform Notes
- Tested on macOS (Apple Silicon) and Linux
- The embedding subprocess (`embed_texts.py`) runs separately to avoid torch/FAISS BLAS conflicts on Apple Silicon
- FAISS CPU version is used by default; GPU acceleration is not required

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rss_agent_claude.git
cd rss_agent_claude

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Edit .env and add your Anthropic API key
```

## Configuration

The pipeline is fully customizable through config files in the `config/` directory. Example configs are provided in `config/examples/` — copy them to `config/` and edit to match your research interests.

```bash
# Copy example configs as a starting point
cp config/examples/feeds.txt config/feeds.txt
cp config/examples/interest_profile.txt config/interest_profile.txt
cp config/examples/ranking_prompt.txt config/ranking_prompt.txt
cp config/examples/crossref_terms.txt config/crossref_terms.txt
cp config/examples/biorxiv_terms.txt config/biorxiv_terms.txt
cp config/examples/biorxiv_subjects.txt config/biorxiv_subjects.txt
cp config/examples/crossref_issn_fallbacks.txt config/crossref_issn_fallbacks.txt
```

### Config Files

| File | Purpose |
|------|---------|
| `feeds.txt` | RSS/Atom feed URLs, one per line |
| `interest_profile.txt` | Short list of your research interests (drives semantic filtering) |
| `ranking_prompt.txt` | Detailed LLM scoring prompt with your interests, hard negatives, and rubric |
| `crossref_terms.txt` | Broad search terms for Crossref supplemental fetching |
| `biorxiv_terms.txt` | Keyword filter for bioRxiv preprints (substring match on title+abstract) |
| `biorxiv_subjects.txt` | bioRxiv subject categories to monitor |
| `crossref_issn_fallbacks.txt` | Journal ISSNs for direct Crossref queries (useful when RSS feeds are blocked) |

### Customizing for Your Research

The most important files to customize are:

#### 1. `interest_profile.txt`
A simple list of your research topics, one per line. This drives the semantic embedding filter. Be specific enough to distinguish your field but broad enough to catch relevant work:

```
Protein structure prediction
Computational drug design
Molecular dynamics simulation
Machine learning for chemistry
```

#### 2. `ranking_prompt.txt`
This is the heart of the system — the prompt that tells Claude how to score each article. Structure it with:

- **Domain expertise statement** — what fields you work in
- **High-priority interests** — topics you'd definitely want to read about
- **Medium-priority interests** — supporting or enabling topics
- **Hard negatives** — topics that use similar vocabulary but aren't relevant to you
- **Scoring rubric** — consistent scale from 0 to 1
- **Calibration notes** — edge cases and guidance for the scorer

The hard negatives section is especially important — without it, the LLM may score tangentially related papers too high.

A real-world example is included: `config/examples/ranking_prompt_marine_microbiology.txt` is the production prompt that I've honed in on over many iterations and represents my rubric as of this posting (marine microbial ecology / microbial physiology). Comparing it to the generic `ranking_prompt.txt` example shows how to adapt the structure for your own field.

#### 3. `feeds.txt`
Find RSS/Atom feeds for journals in your field. Most publishers provide them:
- Check journal homepages for RSS icons
- Try `https://journalsite.com/rss/` or `/feed/`
- Use the built-in `discover_feeds.py` utility to search for feeds
- Run `validate_feeds.py` periodically to check for broken feeds

#### 4. `crossref_terms.txt` and `biorxiv_terms.txt`
These cast a wider net beyond your RSS feeds:
- Crossref terms should be broader (they're searched as full queries)
- bioRxiv terms are substring-matched against title+abstract, so they can be more specific

### Optional: Personal Paper Library

If you have a collection of papers you've already read (e.g., exported from a reference manager as JSON), you can build a FAISS index from it to improve novelty scoring:

```bash
# Export your library as JSON with title and abstract fields
# Place it at library/source_papers/paperpile_library.json

python scripts/build_library_index.py
```

This creates an embedding index of your existing library. The pipeline uses it to score how *novel* each new article is — papers similar to ones you've already read get a lower novelty score, helping surface truly new work.

Without a personal library, the novelty component is skipped and scoring relies on the semantic filter and LLM ranking alone.

## Usage

### Daily Run

```bash
# Run the full pipeline
bash scripts/run_pipeline.sh

# Outputs appear in:
#   data/outputs/markdown/   — Markdown digest
#   data/outputs/html/       — HTML digest (styled, suitable for email)
#   data/outputs/csv/        — CSV export
```

### Tuning Parameters

All major parameters can be overridden via environment variables:

```bash
# Limit LLM API calls (default: 2000 articles max)
MAX_LLM=500 bash scripts/run_pipeline.sh

# Raise the minimum score threshold for the digest
MIN_SCORE=0.65 bash scripts/run_pipeline.sh

# Skip bioRxiv fetching (faster runs)
SKIP_BIORXIV=1 bash scripts/run_pipeline.sh

# Adjust how far back Crossref looks
CROSSREF_DAYS_BACK=14 bash scripts/run_pipeline.sh
```

#### Key Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_SCORE` | `0.60` | Minimum final score to include in the digest |
| `TOP_PICKS_SCORE` | `0.75` | Score threshold for the "Top Picks" tier |
| `MAX_LLM` | `2000` | Maximum articles to send to the LLM for scoring |
| `SEMANTIC_THRESHOLD` | `0.25` | Minimum semantic similarity to pass pre-filter |
| `MAX_AGE_DAYS` | `3` | Only process articles newer than this |
| `CROSSREF_DAYS_BACK` | `21` | How far back Crossref queries look |
| `FETCH_WORKERS` | `4` | Parallel feed-fetching workers |
| `RANK_ONLY_NEW` | `1` | Only rank articles not seen in previous runs |
| `SKIP_CROSSREF` | `0` | Set to `1` to disable Crossref supplementation |
| `SKIP_BIORXIV` | `0` | Set to `1` to disable bioRxiv fetching |
| `SKIP_NATURE` | `0` | Set to `1` to disable Nature journal fetching |
| `SKIP_ASM` | `0` | Set to `1` to disable ASM Press fetching |
| `NOVEL_THRESHOLD` | `0.60` | Minimum novelty score (only matters with a library index) |
| `VALIDATE_FEEDS` | `0` | Set to `1` to validate feeds before fetching |
| `RESET_DIGESTED` | `1` | Set to clear the "already shown" history |

### Automation

#### cron (Linux/macOS)

```bash
# Run daily at 6:00 AM
0 6 * * * cd /path/to/rss_agent_claude && bash scripts/run_pipeline.sh >> data/logs/pipeline.log 2>&1
```

#### launchd (macOS)

Create `~/Library/LaunchAgents/com.rss-agent-claude.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.rss-agent-claude</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/path/to/rss_agent_claude/scripts/run_pipeline.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>6</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>WorkingDirectory</key>
    <string>/path/to/rss_agent_claude</string>
    <key>StandardOutPath</key>
    <string>/path/to/rss_agent_claude/data/logs/pipeline.log</string>
    <key>StandardErrorPath</key>
    <string>/path/to/rss_agent_claude/data/logs/pipeline.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.rss-agent-claude.plist
```

### Evaluation

If you share articles on Bluesky, you can evaluate the pipeline against your posting history:

```bash
# Set your Bluesky handle
export BLUESKY_HANDLE=yourhandle.bsky.social

# Run evaluation
python scripts/eval_pipeline.py --days-back 7

# Or pass the handle directly
python scripts/eval_pipeline.py --handle yourhandle.bsky.social --days-back 7
```

This shows precision/recall metrics and diagnoses where missed articles were lost in the pipeline (not fetched, filtered out, or scored too low).

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `validate_feeds.py` | Check all feeds for errors, empty responses, or timeouts |
| `discover_feeds.py` | Search for RSS feeds for a given journal or topic |
| `find_replacement_feeds.py` | Find alternative feeds for broken ones |
| `prune_cache.py` | Remove old articles from the cache (run automatically) |
| `build_library_index.py` | Build FAISS index from your paper library |
| `discover_topics.py` | Discover topic clusters in your library |
| `classify_new_topics.py` | Classify new articles into discovered topics |

## Project Structure

```
rss_agent_claude/
├── scripts/
│   ├── run_pipeline.sh          # Main orchestration script
│   ├── fetch_feeds.py           # Multi-source article fetching
│   ├── filter_and_rank.py       # Semantic filtering + LLM ranking
│   ├── generate_markdown.py     # Digest output generation
│   ├── embed_texts.py           # Embedding subprocess (torch isolation)
│   ├── common_paths.py          # Shared path constants
│   ├── eval_pipeline.py         # Precision/recall evaluation
│   ├── validate_feeds.py        # Feed health checker
│   ├── prune_cache.py           # Cache cleanup
│   ├── build_library_index.py   # Personal library FAISS index builder
│   ├── discover_feeds.py        # Feed discovery utility
│   ├── discover_topics.py       # Topic clustering utility
│   ├── classify_new_topics.py   # Topic classification utility
│   └── find_replacement_feeds.py# Feed replacement finder
├── config/
│   ├── examples/                # Example configs (copy to config/ and customize)
│   │   ├── feeds.txt
│   │   ├── interest_profile.txt
│   │   ├── ranking_prompt.txt
│   │   ├── crossref_terms.txt
│   │   ├── biorxiv_terms.txt
│   │   ├── biorxiv_subjects.txt
│   │   └── crossref_issn_fallbacks.txt
│   ├── feeds.txt                # Your RSS feed list
│   ├── interest_profile.txt     # Your research interests
│   └── ranking_prompt.txt       # Your LLM scoring prompt
├── data/
│   ├── cache/                   # Pipeline state and intermediate data
│   ├── outputs/                 # Generated digests (markdown, html, csv)
│   ├── logs/                    # Pipeline run logs
│   └── tmp/                     # Temporary files
├── library/                     # Personal paper library (optional)
│   ├── embeddings/              # Library embedding vectors
│   ├── faiss/                   # FAISS index files
│   ├── source_papers/           # Source library JSON
│   └── topics/                  # Topic model data
├── .env.example                 # Environment variable template
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## API Costs

The primary cost is Claude Haiku API calls for article ranking. With prompt caching enabled (the ranking prompt is sent as a cached system message), typical costs are:

- **~$0.05–0.15 per daily run** depending on article volume
- The semantic pre-filter reduces LLM calls by 80–90% compared to scoring everything
- Set `MAX_LLM` to cap the maximum number of articles scored per run

The embedding model (`all-mpnet-base-v2`) runs locally — no API cost for semantic filtering.

## Troubleshooting

### Common Issues

**"No module named 'faiss'"**
Install the CPU version: `pip install faiss-cpu`

**Torch/FAISS BLAS conflict on Apple Silicon**
This is handled automatically — `embed_texts.py` runs as a subprocess to isolate torch from faiss. If you still see issues, ensure you're using the versions in `requirements.txt`.

**Empty digest / no articles**
- Check that your `feeds.txt` URLs are valid: `VALIDATE_FEEDS=1 bash scripts/run_pipeline.sh`
- Lower `SEMANTIC_THRESHOLD` (default 0.25) if too many articles are filtered out
- Lower `MIN_SCORE` (default 0.60) if articles pass filtering but score too low
- Check `data/cache/raw_feeds.json` to see what was fetched

**High API costs**
- Reduce `MAX_LLM` to limit scored articles per run
- Set `RANK_ONLY_NEW=1` (default) to avoid re-scoring articles from previous runs
- Disable high-volume sources: `SKIP_BIORXIV=1`, `SKIP_NATURE=1`

**Feed validation errors**
Run `python scripts/validate_feeds.py` to identify broken or empty feeds, then either fix the URLs or use `find_replacement_feeds.py` to locate alternatives.

## License

MIT License — see [LICENSE](LICENSE) for details.

Original `rss_agent` by [John Casey](https://github.com/jrcasey/rss_agent).
