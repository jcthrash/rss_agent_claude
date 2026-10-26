import json
from pathlib import Path
from typing import Any

import pandas as pd
from bertopic import BERTopic

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RANKED_PATH = PROJECT_ROOT / "data/cache/ranked_articles.json"
TOPIC_MODEL_PATH = PROJECT_ROOT / "library/topics/bertopic_model"
TOPIC_INFO_PATH = PROJECT_ROOT / "library/topics/topic_info.csv"
NEW_IDS_PATH = PROJECT_ROOT / "data/cache/new_ids.json"


def simplify_topic_label(raw_name: str, topic_id: int) -> str:
    if not raw_name or not isinstance(raw_name, str):
        return f"Topic {topic_id}"

    parts = [p for p in raw_name.split("_") if p]
    if parts and parts[0].isdigit():
        parts = parts[1:]

    stopwords = {
        "study", "analysis", "using", "based", "associated", "new", "role",
        "effects", "effect", "responses", "response", "approach", "approaches",
        "microbial", "bacterial", "community", "communities", "organisms", "organism",
    }

    cleaned = []
    for p in parts:
        p = p.strip().lower()
        if not p or p in stopwords or len(p) <= 2:
            continue
        cleaned.append(p)

    if not cleaned:
        return f"Topic {topic_id}"

    return " ".join(cleaned[:4]).title()


def build_text(article: dict[str, Any]) -> str:
    return f"{article.get('title', '')}. {article.get('summary', '') or article.get('abstract', '') or article.get('content', '')}".strip()


def main() -> None:
    if not RANKED_PATH.exists():
        print("No ranked articles to classify.")
        return

    articles = json.loads(RANKED_PATH.read_text(encoding="utf-8"))
    if not articles:
        print("Ranked article list empty.")
        return

    print("Loading BERTopic model...")
    topic_model = BERTopic.load(str(TOPIC_MODEL_PATH))

    topic_names: dict[int, str] = {}
    if TOPIC_INFO_PATH.exists():
        df = pd.read_csv(TOPIC_INFO_PATH)
        topic_names = {int(row.Topic): str(row.Name) for _, row in df.iterrows()}

    new_ids = set()
    if NEW_IDS_PATH.exists():
        try:
            new_ids = set(json.loads(NEW_IDS_PATH.read_text(encoding="utf-8")))
        except Exception:
            new_ids = set()

    candidates = []
    for article in articles:
        if article.get("topic") is not None and article.get("topic_label"):
            continue
        if new_ids and article.get("id") not in new_ids and article.get("topic") is not None:
            continue
        candidates.append(article)

    if not candidates:
        print("No articles need topic classification.")
        return

    texts = [build_text(a) for a in candidates]
    print(f"Classifying {len(texts)} articles...")
    topics, _ = topic_model.transform(texts)

    article_topic_map = {id(article): int(topic) for article, topic in zip(candidates, topics)}
    for article in articles:
        if id(article) not in article_topic_map:
            continue
        topic = article_topic_map[id(article)]
        article["topic"] = topic
        raw_name = topic_names.get(topic, f"Topic {topic}")
        article["topic_label"] = simplify_topic_label(raw_name, topic)

    RANKED_PATH.write_text(json.dumps(articles, indent=2), encoding="utf-8")
    print("Topics assigned and saved.")


if __name__ == "__main__":
    main()
