#!/usr/bin/env python3
"""
Build a FAISS embedding index from the Paperpile library JSON.

Embedding model: sentence-transformers all-mpnet-base-v2 (local, no API cost).
Runs on Apple Silicon MPS automatically when available via PyTorch.

Produces:
  library/embeddings/library_metadata.json   — normalised paper records
  library/embeddings/library_embeddings.npy  — float32 embedding matrix
  library/faiss/library_faiss.index          — FAISS IndexFlatIP index
  library/faiss/id_map.json                  — integer index → record index map

Checkpointing: saves progress every CHECKPOINT_EVERY_BATCHES batches so a
long run can be resumed if interrupted.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from common_paths import (
    CHECKPOINT_EMBEDDINGS_NPY,
    CHECKPOINT_METADATA_JSON,
    CHECKPOINT_STATE_JSON,
    LIBRARY_EMBEDDINGS_NPY,
    LIBRARY_FAISS_INDEX,
    LIBRARY_ID_MAP_JSON,
    LIBRARY_METADATA_JSON,
    PAPERPILE_LIBRARY_JSON,
    ensure_project_dirs,
    load_json,
    save_json,
)

MODEL_NAME             = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE             = 32
CHECKPOINT_EVERY_BATCHES = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a FAISS embedding index from the Paperpile library JSON"
    )
    p.add_argument("--input-json",       type=Path, default=PAPERPILE_LIBRARY_JSON)
    p.add_argument("--metadata-out",     type=Path, default=LIBRARY_METADATA_JSON)
    p.add_argument("--embeddings-out",   type=Path, default=LIBRARY_EMBEDDINGS_NPY)
    p.add_argument("--index-out",        type=Path, default=LIBRARY_FAISS_INDEX)
    p.add_argument("--id-map-out",       type=Path, default=LIBRARY_ID_MAP_JSON)
    p.add_argument("--checkpoint-meta",  type=Path, default=CHECKPOINT_METADATA_JSON)
    p.add_argument("--checkpoint-emb",   type=Path, default=CHECKPOINT_EMBEDDINGS_NPY)
    p.add_argument("--checkpoint-state", type=Path, default=CHECKPOINT_STATE_JSON)
    p.add_argument("--batch-size",       type=int,  default=BATCH_SIZE)
    return p.parse_args()


# ── Record helpers ─────────────────────────────────────────────────────────────

def build_text(record: dict[str, Any]) -> str:
    title    = (record.get("title")    or "").strip()
    abstract = (record.get("abstract") or "").strip()
    return f"Title: {title}\nAbstract: {abstract}".strip()


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    title = record.get("title")
    if not title:
        return {}
    published = record.get("published") or {}
    urls = record.get("url") or []
    return {
        "title":    title,
        "abstract": record.get("abstract"),
        "journal":  record.get("journal"),
        "year":     published.get("year"),
        "doi":      record.get("doi"),
        "url":      urls,
        "text":     build_text(record),
    }


def load_paperpile_json(path: Path) -> list[dict[str, Any]]:
    data = load_json(path, None)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of paper records in {path}")
    return data


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(
    meta_path: Path,
    emb_path: Path,
    state_path: Path,
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    next_index: int,
) -> None:
    save_json(meta_path, records)
    np.save(emb_path, embeddings)
    save_json(state_path, {"next_index": next_index})


def load_checkpoint(
    meta_path: Path, emb_path: Path, state_path: Path
) -> tuple[list[dict[str, Any]], np.ndarray, int]:
    if not (meta_path.exists() and emb_path.exists() and state_path.exists()):
        return [], np.empty((0, 0), dtype="float32"), 0
    records    = load_json(meta_path, [])
    embeddings = np.load(emb_path)
    state      = load_json(state_path, {"next_index": 0})
    return records, embeddings, int(state.get("next_index", 0))


def clear_checkpoints(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_project_dirs()
    args = parse_args()

    # Disable tokenizer parallelism to prevent segfault on macOS
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load and normalise records
    raw = load_paperpile_json(args.input_json)
    records = [nr for r in raw if (nr := normalize_record(r)) and len(nr["text"]) >= 20]
    print(f"Papers loaded: {len(records)}")

    # Load model via transformers directly (avoids sentence-transformers
    # multiprocessing machinery that segfaults on some Apple Silicon setups)
    print(f"Loading embedding model: {MODEL_NAME}")
    device = torch.device("cpu")
    # use_fast=False: avoids Rust-based tokenizer parallelism that segfaults
    # on macOS when combined with Python's fork-based multiprocessing
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"  Device: {device}")

    # Resume from checkpoint if available
    checkpoint_records, checkpoint_embeddings, start_idx = load_checkpoint(
        args.checkpoint_meta, args.checkpoint_emb, args.checkpoint_state
    )
    if checkpoint_records and len(checkpoint_records) != len(records):
        print("Checkpoint record count mismatch — starting fresh.")
        checkpoint_records, checkpoint_embeddings, start_idx = [], np.empty((0, 0), dtype="float32"), 0

    if start_idx > 0 and checkpoint_embeddings.size > 0:
        print(f"Resuming from checkpoint at record {start_idx}...")
        all_vecs = [checkpoint_embeddings]
    else:
        print("Starting fresh embedding run...")
        all_vecs  = []
        start_idx = 0

    # Embed in batches with checkpointing
    batch_counter = 0
    texts_to_embed = [r["text"] for r in records]

    def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    def embed_texts(texts: list[str]) -> np.ndarray:
        # use_fast=False avoids Rust tokenizer parallelism that segfaults on macOS
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=512,
                           return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            out = model(**encoded)
        vecs = mean_pool(out.last_hidden_state, encoded["attention_mask"])
        vecs = F.normalize(vecs, p=2, dim=1)
        return vecs.cpu().numpy().astype("float32")

    total_batches = (len(records) - start_idx + args.batch_size - 1) // args.batch_size
    for i in range(start_idx, len(records), args.batch_size):
        batch_num = (i - start_idx) // args.batch_size + 1
        if batch_num == 1 or batch_num % 10 == 0 or i + args.batch_size >= len(records):
            print(f"  Embedding batch {batch_num}/{total_batches} (record {i}/{len(records)})")
        batch_texts = texts_to_embed[i : i + args.batch_size]
        vecs = embed_texts(batch_texts)
        all_vecs.append(vecs)
        batch_counter += 1
        next_index = i + len(batch_texts)

        if batch_counter % CHECKPOINT_EVERY_BATCHES == 0:
            save_checkpoint(
                args.checkpoint_meta, args.checkpoint_emb, args.checkpoint_state,
                records, np.vstack(all_vecs), next_index,
            )

    embeddings = np.vstack(all_vecs) if all_vecs else np.empty((0, 0), dtype="float32")
    if embeddings.size == 0:
        raise RuntimeError("No embeddings were generated.")

    print(f"Embedding matrix: {embeddings.shape}  (dtype={embeddings.dtype})")

    # Import faiss here (after torch is done) to avoid BLAS conflict on Apple Silicon
    import faiss

    # Build FAISS index (IndexFlatIP = cosine similarity on normalised vectors)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save outputs
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.metadata_out, records)
    np.save(args.embeddings_out, embeddings)
    faiss.write_index(index, str(args.index_out))
    save_json(args.id_map_out, {str(i): i for i in range(len(records))})

    clear_checkpoints(args.checkpoint_meta, args.checkpoint_emb, args.checkpoint_state)

    print(f"Saved metadata    → {args.metadata_out}")
    print(f"Saved embeddings  → {args.embeddings_out}")
    print(f"Saved FAISS index → {args.index_out}")
    print(f"Saved id map      → {args.id_map_out}")


if __name__ == "__main__":
    main()
