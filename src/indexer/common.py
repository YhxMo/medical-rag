"""Shared persistence and hybrid-ranking helpers for evidence stores."""
from __future__ import annotations

import json
import logging
import pickle
import re
from functools import lru_cache
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.schema import EvidenceItem, RetrievalHit


DEFAULT_EVIDENCE_TYPE_WEIGHTS = {
    "text": 1.0,
    "image_caption": 1.0,
    "exercise_qa": 0.92,
}
_RRF_K = 60


def retrieval_text(item: EvidenceItem) -> str:
    """Only versioned enhanced records opt into chapter-aware retrieval."""
    heading = item.metadata.get("retrieval_heading", "")
    return f"{heading}\n{item.content}" if heading else item.content


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
            handle.write(json.dumps(evidence_to_json(item), ensure_ascii=False) + "\n")


def read_evidence(path: Path) -> list[EvidenceItem]:
    """Load canonical evidence records from JSON Lines."""
    evidence: list[EvidenceItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                evidence.append(evidence_from_json(json.loads(line)))
    return evidence


def build_bm25_index(path: Path, evidence: list[EvidenceItem]) -> None:
    """Build the sparse sidecar index shared by both vector backends."""
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(retrieval_text(item)) for item in evidence]
    bm25 = BM25Okapi(tokenized)
    with path.open("wb") as handle:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, handle)


def bm25_candidates(path: Path, query: str, *, limit: int) -> list[tuple[int, float]]:
    """Return BM25 candidates as (evidence ordinal, score) pairs."""
    if limit <= 0:
        return []

    stat = path.stat()
    bm25 = _cached_bm25(str(path.resolve()), (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    scores = bm25.get_scores(tokenize(query))
    order = np.argsort(scores)[::-1][:limit]
    return [(int(index), float(scores[index])) for index in order]


def fuse_hybrid_candidates(
    evidence: list[EvidenceItem],
    *,
    dense: Iterable[tuple[int, float]],
    bm25: Iterable[tuple[int, float]],
    final_top_k: int,
    evidence_type_weights: dict[str, float] | None = None,
) -> list[RetrievalHit]:
    """Fuse dense and BM25 rankings with RRF and evidence-type weighting."""
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

    type_weights = evidence_type_weights or DEFAULT_EVIDENCE_TYPE_WEIGHTS
    for ordinal, scores in fused.items():
        evidence_type = evidence[ordinal].evidence_type
        scores["weighted"] = scores["rrf"] * float(type_weights.get(evidence_type, 1.0))

    ordered = sorted(fused.items(), key=lambda item: item[1]["weighted"], reverse=True)[:final_top_k]
    return [
        RetrievalHit(
            evidence=evidence[ordinal],
            score=scores["weighted"],
            rank=rank,
            scores=scores,
        )
        for rank, (ordinal, scores) in enumerate(ordered, start=1)
    ]


def tokenize(text: str) -> list[str]:
    """Tokenize Chinese text for BM25, with a dependency-light fallback."""
    if not re.search(r"[\u4e00-\u9fff]", text):
        return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text.casefold())
    try:
        import jieba

        jieba.setLogLevel(logging.WARNING)
        return [token.strip() for token in jieba.lcut(text) if token.strip()]
    except ImportError:
        return [text[index : index + 2] for index in range(max(0, len(text) - 1))]


def evidence_to_json(item: EvidenceItem) -> dict[str, Any]:
    return asdict(item)


def evidence_from_json(data: dict[str, Any]) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=data["evidence_id"],
        evidence_type=data["evidence_type"],
        source_file=data["source_file"],
        content=data["content"],
        page_start=data.get("page_start"),
        page_end=data.get("page_end"),
        metadata=data.get("metadata", {}),
    )
