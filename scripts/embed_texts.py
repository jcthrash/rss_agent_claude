#!/usr/bin/env python3
"""
embed_texts.py  —  standalone embedder, called as a subprocess by filter_and_rank.py

Reads texts from a JSON file, embeds them with all-mpnet-base-v2, writes
float32 numpy array to an .npy file, then exits.

This runs in a separate process so that torch and faiss never coexist in the
same address space (they conflict on Apple Silicon / PyTorch 2.11+).

Usage (internal — called by filter_and_rank.py):
    python scripts/embed_texts.py --texts /tmp/texts.json --out /tmp/vecs.npy
"""
from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME  = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE  = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--texts", type=Path, required=True, help="JSON file containing list of strings")
    p.add_argument("--out",   type=Path, required=True, help="Output .npy file for embeddings")
    return p.parse_args()


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def main() -> None:
    args = parse_args()

    texts = json.loads(args.texts.read_text(encoding="utf-8"))
    if not texts:
        np.save(args.out, np.empty((0, 768), dtype="float32"))
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model     = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    all_vecs = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(texts), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        if batch_num == 1 or batch_num % 50 == 0 or batch_num == total_batches:
            print(f"  [{batch_num}/{total_batches}] embedding texts {i}-{min(i+BATCH_SIZE, len(texts))}...", file=sys.stderr, flush=True)
        batch = texts[i : i + BATCH_SIZE]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            max_length=512, return_tensors="pt")
        with torch.no_grad():
            out = model(**encoded)
        vecs = mean_pool(out.last_hidden_state, encoded["attention_mask"])
        vecs = F.normalize(vecs, p=2, dim=1)
        all_vecs.append(vecs.cpu().detach().numpy().astype("float32"))
        del out, vecs, encoded

    result = np.vstack(all_vecs)
    np.save(args.out, result)
    print(f"Embedded {len(texts)} texts → shape {result.shape}", file=sys.stderr)


if __name__ == "__main__":
    main()
