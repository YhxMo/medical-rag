"""Evaluation runner for labelled retrieval metrics."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.embedding.base import EmbeddingProvider
from src.evaluation.dataset import EvaluationQuestion
from src.indexer.qdrant_store import QdrantIndexStore
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
    top_k: int = 5
    search_options: dict[str, Any] = field(default_factory=dict)

    @property
    def hit_rate_at_k(self) -> float:
        return self.hit_count / self.total if self.total else 0.0


def evaluate_retrieval(
    questions: list[EvaluationQuestion],
    store: QdrantIndexStore,
    embedding_provider: EmbeddingProvider,
    *,
    top_k: int = 5,
    reranker: Reranker | None = None,
    search_kwargs: dict[str, Any] | None = None,
) -> EvaluationResult:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not questions:
        raise ValueError("Evaluation requires at least one question")
    if any(not q.expected_evidence_ids for q in questions):
        raise ValueError("Every evaluation question requires relevance labels")
    reranker = reranker or NoopReranker()
    search_options = dict(search_kwargs or {})
    search_options.setdefault("final_top_k", top_k)
    details: list[dict] = []

    for question in questions:
        retrieved = store.search(question.question, embedding_provider, **search_options)
        # Deduplicate before the cutoff so repeated evidence cannot inflate metrics.
        retrieved = reranker.rerank(question.question, retrieved, top_k=None)
        retrieved_ids = list(dict.fromkeys(hit.evidence.evidence_id for hit in retrieved))[:top_k]
        expected = set(question.expected_evidence_ids)

        is_hit = bool(expected and expected.intersection(retrieved_ids))

        matches = sum(1 for evidence_id in retrieved_ids if evidence_id in expected)
        precision = matches / top_k
        recall = matches / len(expected)

        reciprocal_rank = 0.0
        for rank, evidence_id in enumerate(retrieved_ids, start=1):
            if evidence_id in expected:
                reciprocal_rank = 1.0 / rank
                break

        ndcg = _ndcg(retrieved_ids, expected, top_k)

        details.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "expected_evidence_ids": list(question.expected_evidence_ids),
                "retrieved_evidence_ids": retrieved_ids,
                "hit": is_hit,
                "recall": recall,
                "precision": precision,
                "reciprocal_rank": reciprocal_rank,
                "ndcg": ndcg,
            }
        )

    total = len(questions)
    return EvaluationResult(
        total=total,
        recall_at_k=sum(d["recall"] for d in details) / total,
        hit_count=sum(d["hit"] for d in details),
        precision_at_k=sum(d["precision"] for d in details) / total,
        mrr=sum(d["reciprocal_rank"] for d in details) / total,
        ndcg_at_k=sum(d["ndcg"] for d in details) / total,
        details=details,
        top_k=top_k,
        search_options=search_options,
    )


def _ndcg(retrieved_ids: list[str], expected: set[str], top_k: int) -> float:
    """Binary-relevance NDCG over the retrieved ranking."""
    if not retrieved_ids or not expected:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, evidence_id in enumerate(retrieved_ids, start=1)
        if evidence_id in expected
    )
    ideal_hits = min(len(expected), top_k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def result_summary(result: EvaluationResult) -> dict[str, Any]:
    """Allowlisted aggregate report: never serialize questions, IDs or config."""
    return {
        "schema_version": 2,
        "metrics_version": "retrieval-v2",
        "top_k": result.top_k,
        "search_options": {
            key: int(result.search_options[key])
            for key in ("dense_top_k", "bm25_top_k", "final_top_k")
            if key in result.search_options
        },
        "total": result.total,
        "hit_count": result.hit_count,
        "hit_rate_at_k": result.hit_rate_at_k,
        "recall_at_k": result.recall_at_k,
        "precision_at_k": result.precision_at_k,
        "mrr": result.mrr,
        "ndcg_at_k": result.ndcg_at_k,
    }


def save_result(path: str | Path, result: EvaluationResult, *, inputs=None) -> None:
    """Keep aggregate and per-question results together; never overwrite a previous run."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"metrics": result_summary(result), "details": result.details, "inputs": inputs or {}}
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
