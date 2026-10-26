#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import json
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from common_paths import BERTOPIC_MODEL_DIR, DOCUMENT_TOPICS_JSON, LIBRARY_METADATA_JSON, TOPIC_INFO_CSV, ensure_project_dirs

EMBED_MODEL = 'all-MiniLM-L6-v2'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Fit BERTopic on library metadata abstracts/titles')
    p.add_argument('--library-meta', type=Path, default=LIBRARY_METADATA_JSON)
    p.add_argument('--model-dir', type=Path, default=BERTOPIC_MODEL_DIR)
    p.add_argument('--doc-topics-json', type=Path, default=DOCUMENT_TOPICS_JSON)
    p.add_argument('--topic-info-csv', type=Path, default=TOPIC_INFO_CSV)
    return p.parse_args()


def load_library_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}')
    return json.loads(path.read_text())


def make_docs(records: list[dict[str, Any]]) -> list[str]:
    docs = []
    for r in records:
        title = (r.get('title') or '').strip()
        abstract = (r.get('abstract') or '').strip()
        text = f'{title}. {abstract}'.strip()
        docs.append(text if len(text) > 20 else (title or 'Untitled'))
    return docs


def main() -> None:
    ensure_project_dirs()
    args = parse_args()
    records = load_library_metadata(args.library_meta)
    docs = make_docs(records)
    print(f'Loaded {len(docs)} library documents')
    embedding_model = SentenceTransformer(EMBED_MODEL)
    topic_model = BERTopic(embedding_model=embedding_model, verbose=True, calculate_probabilities=False, min_topic_size=25)
    topics, _ = topic_model.fit_transform(docs)
    topic_model.save(str(args.model_dir))
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(args.topic_info_csv, index=False)
    out = []
    for rec, topic in zip(records, topics):
        out.append({'title': rec.get('title'), 'journal': rec.get('journal'), 'year': rec.get('year'), 'doi': rec.get('doi'), 'topic': int(topic)})
    args.doc_topics_json.write_text(json.dumps(out, indent=2))
    print(f'Saved BERTopic model to {args.model_dir}')


if __name__ == '__main__':
    main()
