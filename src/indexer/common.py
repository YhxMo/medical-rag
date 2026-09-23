"""Persistence, BM25 tokenization and reciprocal rank fusion."""

from __future__ import annotations

import json
import logging
import pickle
import re
from collections.abc import Iterable
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.schema import EvidenceItem, RetrievalHit

_RRF_K = 60


@lru_cache(maxsize=8)
def _cached_bm25(path: str, version: tuple[int, int, int, int]):
    # Local, trusted build artifact only. stat identity invalidates replacements.
    with open(path, "rb") as handle:
        return pickle.load(handle)["bm25"]


def write_evidence(path: Path, evidence: list[EvidenceItem]) -> None:
    """Persist canonical evidence records for BM25, evaluation, and inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in evidence:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def read_evidence(path: Path) -> list[EvidenceItem]:
    """Load canonical evidence records from JSON Lines."""
    evidence: list[EvidenceItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                evidence.append(EvidenceItem(**json.loads(line)))
    return evidence


def build_bm25_index(path: Path, evidence: list[EvidenceItem]) -> None:
    """Build the BM25 sidecar index."""
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(item.content) for item in evidence]
    bm25 = BM25Okapi(tokenized)
    with path.open("wb") as handle:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, handle)


def bm25_candidates(path: Path, query: str, *, limit: int) -> list[tuple[int, float]]:
    """Return BM25 candidates as (evidence ordinal, score) pairs."""
    if limit <= 0:
        return []

    stat = path.stat()
    bm25 = _cached_bm25(
        str(path.resolve()), (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    )
    scores = bm25.get_scores(tokenize(query))
    order = np.argsort(scores)[::-1][:limit]
    return [(int(index), float(scores[index])) for index in order]


def fuse_hybrid_candidates(
    evidence: list[EvidenceItem],
    *,
    dense: Iterable[tuple[int, float]],
    bm25: Iterable[tuple[int, float]],
    final_top_k: int,
) -> list[RetrievalHit]:
    """Fuse dense and BM25 rankings with reciprocal rank fusion."""
    fused: dict[int, dict[str, float]] = {}

    for rank, (ordinal, score) in enumerate(dense, start=1):
        if ordinal < 0 or ordinal >= len(evidence):
            continue
        fused.setdefault(ordinal, {"dense": 0.0, "bm25": 0.0, "rrf": 0.0})
        fused[ordinal]["dense"] = float(score)
        fused[ordinal]["rrf"] += 1.0 / (_RRF_K + rank)

    for rank, (ordinal, score) in enumerate(bm25, start=1):
        if ordinal < 0 or ordinal >= len(evidence):
            continue
        fused.setdefault(ordinal, {"dense": 0.0, "bm25": 0.0, "rrf": 0.0})
        fused[ordinal]["bm25"] = float(score)
        fused[ordinal]["rrf"] += 1.0 / (_RRF_K + rank)

    ordered = sorted(fused.items(), key=lambda item: item[1]["rrf"], reverse=True)[:final_top_k]
    return [
        RetrievalHit(
            evidence=evidence[ordinal],
            score=scores["rrf"],
            rank=rank,
            scores=scores,
        )
        for rank, (ordinal, scores) in enumerate(ordered, start=1)
    ]


def tokenize(text: str) -> list[str]:
    """Tokenize English terms and Chinese words for BM25."""
    if not re.search(r"[\u4e00-\u9fff]", text):
        return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text.casefold())
    import jieba

    jieba.setLogLevel(logging.WARNING)
    return [token.strip() for token in jieba.lcut(text) if token.strip()]
