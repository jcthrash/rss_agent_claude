from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / 'config'
DATA_DIR = PROJECT_ROOT / 'data'
CACHE_DIR = DATA_DIR / 'cache'
OUTPUT_DIR = DATA_DIR / 'outputs'
LOG_DIR = DATA_DIR / 'logs'
TMP_DIR = DATA_DIR / 'tmp'
LIBRARY_DIR = PROJECT_ROOT / 'library'
FAISS_DIR = LIBRARY_DIR / 'faiss'
EMBEDDINGS_DIR = LIBRARY_DIR / 'embeddings'
TOPICS_DIR = LIBRARY_DIR / 'topics'
SOURCE_PAPERS_DIR = LIBRARY_DIR / 'source_papers'

RAW_FEEDS_JSON = CACHE_DIR / 'raw_feeds.json'
FILTERED_JSON = CACHE_DIR / 'filtered_articles.json'
RANKED_JSON = CACHE_DIR / 'ranked_articles.json'
SEMANTIC_SCORES_JSON = CACHE_DIR / 'semantic_scores.json'
LLM_SCORES_JSON = CACHE_DIR / 'llm_scores.json'
EMBEDDINGS_CACHE_JSON = CACHE_DIR / 'embeddings.json'
SEEN_IDS_TXT = CACHE_DIR / 'seen_ids.txt'
NEW_IDS_JSON = CACHE_DIR / 'new_ids.json'
NEW_PUBLISHED_IDS_JSON = CACHE_DIR / 'new_published_ids.json'
NEW_HASHES_JSON = CACHE_DIR / 'new_hashes.json'
FETCH_STATE_JSON = CACHE_DIR / 'fetch_state.json'
SEMANTIC_FILTER_STATE_JSON = CACHE_DIR / 'semantic_filter_state.json'
PROFILE_EMBEDDING_JSON = CACHE_DIR / 'profile_embedding.json'
NOVELTY_SCORES_JSON = CACHE_DIR / 'novelty_scores.json'

MARKDOWN_DIR = OUTPUT_DIR / 'markdown'
HTML_DIR = OUTPUT_DIR / 'html'
CSV_DIR = OUTPUT_DIR / 'csv'
REPORTS_DIR = OUTPUT_DIR / 'reports'

FEEDS_TXT = CONFIG_DIR / 'feeds.txt'
INTEREST_PROFILE_TXT = CONFIG_DIR / 'interest_profile.txt'
RANKING_PROMPT_TXT = CONFIG_DIR / 'ranking_prompt.txt'

LIBRARY_METADATA_JSON = EMBEDDINGS_DIR / 'library_metadata.json'
LIBRARY_EMBEDDINGS_NPY = EMBEDDINGS_DIR / 'library_embeddings.npy'
LIBRARY_FAISS_INDEX = FAISS_DIR / 'library_faiss.index'
LIBRARY_ID_MAP_JSON = FAISS_DIR / 'id_map.json'
PAPERPILE_LIBRARY_JSON = SOURCE_PAPERS_DIR / 'paperpile_library.json'
BERTOPIC_MODEL_DIR = TOPICS_DIR / 'bertopic_model'
TOPIC_INFO_CSV = TOPICS_DIR / 'topic_info.csv'
DOCUMENT_TOPICS_JSON = TOPICS_DIR / 'document_topics.json'

CHECKPOINT_METADATA_JSON = EMBEDDINGS_DIR / 'checkpoint_metadata.json'
CHECKPOINT_EMBEDDINGS_NPY = EMBEDDINGS_DIR / 'checkpoint_embeddings.npy'
CHECKPOINT_STATE_JSON = EMBEDDINGS_DIR / 'checkpoint_state.json'


def ensure_project_dirs() -> None:
    for path in [
        CONFIG_DIR,
        CACHE_DIR,
        MARKDOWN_DIR,
        HTML_DIR,
        CSV_DIR,
        REPORTS_DIR,
        LOG_DIR,
        TMP_DIR,
        FAISS_DIR,
        EMBEDDINGS_DIR,
        TOPICS_DIR,
        SOURCE_PAPERS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_text(path: Path, default: str = '') -> str:
    try:
        return path.read_text()
    except Exception:
        return default
