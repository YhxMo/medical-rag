"""Evaluation runner for retrieval-focused MVP metrics."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.evaluation.dataset import EvaluationQuestion
from src.indexer.base import EvidenceStore
from src.embedding.base import EmbeddingProvider
from src.reranker.base import NoopReranker, Reranker


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate evaluation metrics."""

    total: int
    recall_at_k: float
    hit_count: int
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    details: list[dict]


def evaluate_retrieval(
    questions: list[EvaluationQuestion],
    store: EvidenceStore,
    embedding_provider: EmbeddingProvider,
    *,
    top_k: int = 5,
    reranker: Reranker | None = None,
    search_kwargs: dict[str, Any] | None = None,
) -> EvaluationResult:
    reranker = reranker or NoopReranker()
    search_options = dict(search_kwargs or {})
    search_options.setdefault("final_top_k", top_k)
    details: list[dict] = []
    hits = 0
    precision_sum = 0.0
    reciprocal_rank_sum = 0.0
    ndcg_sum = 0.0

    for question in questions:
        retrieved = store.search(question.question, embedding_provider, **search_options)
        retrieved = reranker.rerank(question.question, retrieved, top_k=top_k)
        retrieved_ids = [hit.evidence.evidence_id for hit in retrieved]
        expected = set(question.expected_evidence_ids)

        is_hit = bool(expected and expected.intersection(retrieved_ids))
        hits += int(is_hit)

        matches = sum(1 for evidence_id in retrieved_ids if evidence_id in expected)
        precision = matches / len(retrieved_ids) if retrieved_ids else 0.0

        reciprocal_rank = 0.0
        for rank, evidence_id in enumerate(retrieved_ids, start=1):
            if evidence_id in expected:
                reciprocal_rank = 1.0 / rank
                break

        ndcg = _ndcg(retrieved_ids, expected)

        precision_sum += precision
        reciprocal_rank_sum += reciprocal_rank
        ndcg_sum += ndcg

        details.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "expected_evidence_ids": list(question.expected_evidence_ids),
                "retrieved_evidence_ids": retrieved_ids,
                "hit": is_hit,
                "precision": precision,
                "reciprocal_rank": reciprocal_rank,
                "ndcg": ndcg,
            }
        )

    total = len(questions)
    return EvaluationResult(
        total=total,
        recall_at_k=hits / total if total else 0.0,
        hit_count=hits,
        precision_at_k=precision_sum / total if total else 0.0,
        mrr=reciprocal_rank_sum / total if total else 0.0,
        ndcg_at_k=ndcg_sum / total if total else 0.0,
        details=details,
    )


def _ndcg(retrieved_ids: list[str], expected: set[str]) -> float:
    """Binary-relevance NDCG over the retrieved ranking."""
    if not retrieved_ids or not expected:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, evidence_id in enumerate(retrieved_ids, start=1)
        if evidence_id in expected
    )
    ideal_hits = min(len(expected), len(retrieved_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def save_result(path: str | Path, result: EvaluationResult) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, ensure_ascii=False, indent=2)
