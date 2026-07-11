"""Retrieval ablation experiments for offline project reports."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.embedding.base import EmbeddingProvider
from src.evaluation.dataset import EvaluationQuestion
from src.evaluation.evaluator import EvaluationResult, evaluate_retrieval
from src.indexer.base import EvidenceStore
from src.reranker.base import Reranker


@dataclass(frozen=True)
class RetrievalAblationSpec:
    """One retrieval configuration to evaluate."""

    name: str
    label: str
    description: str
    search_kwargs: dict[str, int]
    use_reranker: bool = False
    exclude_evidence_types: tuple[str, ...] = ()


def default_ablation_specs(
    *,
    candidate_top_k: int = 20,
    include_reranker: bool = True,
    include_no_image_caption: bool = True,
) -> list[RetrievalAblationSpec]:
    """Return the standard retrieval ablation suite."""

    dense_only = {"dense_top_k": candidate_top_k, "bm25_top_k": 0, "final_top_k": candidate_top_k}
    bm25_only = {"dense_top_k": 0, "bm25_top_k": candidate_top_k, "final_top_k": candidate_top_k}
    hybrid = {"dense_top_k": candidate_top_k, "bm25_top_k": candidate_top_k, "final_top_k": candidate_top_k}
    specs = [
        RetrievalAblationSpec(
            name="bm25_only",
            label="BM25 only",
            description="Sparse BM25 retrieval only.",
            search_kwargs=bm25_only,
        ),
        RetrievalAblationSpec(
            name="bge_dense_only",
            label="BGE dense only",
            description="Dense vector retrieval only.",
            search_kwargs=dense_only,
        ),
        RetrievalAblationSpec(
            name="bge_bm25_hybrid",
            label="BGE + BM25 hybrid",
            description="RRF fusion of dense vector retrieval and BM25.",
            search_kwargs=hybrid,
        ),
    ]
    if include_reranker:
        specs.append(
            RetrievalAblationSpec(
                name="bge_bm25_hybrid_bge_reranker",
                label="BGE + BM25 hybrid + BGE reranker",
                description="Hybrid retrieval followed by the configured BGE reranker.",
                search_kwargs=hybrid,
                use_reranker=True,
            )
        )
    if include_no_image_caption:
        specs.append(
            RetrievalAblationSpec(
                name="bge_bm25_hybrid_without_image_caption",
                label="Hybrid without image_caption",
                description="Hybrid retrieval after rebuilding a temporary index without image_caption evidence.",
                search_kwargs=hybrid,
                exclude_evidence_types=("image_caption",),
            )
        )
    return specs


def run_retrieval_ablation(
    questions: list[EvaluationQuestion],
    store: EvidenceStore,
    embedding_provider: EmbeddingProvider,
    *,
    reranker: Reranker | None = None,
    top_k: int = 5,
    candidate_top_k: int = 20,
    tmp_index_dir: str | Path = "artifacts/tmp_ablation",
    specs: list[RetrievalAblationSpec] | None = None,
) -> dict[str, Any]:
    """Run retrieval ablations and return a JSON-serializable report."""

    specs = specs or default_ablation_specs(candidate_top_k=candidate_top_k, include_reranker=reranker is not None)
    base_evidence_count = len(store.load_evidence())
    filtered_store_cache: dict[tuple[str, ...], tuple[EvidenceStore, int]] = {}
    runs = []

    for spec in specs:
        active_store = store
        evidence_count = base_evidence_count
        temp_index = None
        if spec.exclude_evidence_types:
            key = tuple(sorted(spec.exclude_evidence_types))
            if key not in filtered_store_cache:
                filtered_store_cache[key] = _build_filtered_store(
                    store,
                    embedding_provider,
                    excluded_types=key,
                    tmp_index_dir=Path(tmp_index_dir),
                )
            active_store, evidence_count = filtered_store_cache[key]
            temp_index = str(active_store.index_dir)

        result = evaluate_retrieval(
            questions,
            active_store,
            embedding_provider,
            top_k=top_k,
            reranker=reranker if spec.use_reranker else None,
            search_kwargs=spec.search_kwargs,
        )
        runs.append(
            {
                "name": spec.name,
                "label": spec.label,
                "description": spec.description,
                "search_kwargs": dict(spec.search_kwargs),
                "use_reranker": spec.use_reranker,
                "excluded_evidence_types": list(spec.exclude_evidence_types),
                "evidence_count": evidence_count,
                "temp_index": temp_index,
                "metrics": _metrics(result, top_k),
                "details": result.details,
            }
        )

    return {
        "top_k": top_k,
        "candidate_top_k": candidate_top_k,
        "question_count": len(questions),
        "base_evidence_count": base_evidence_count,
        "runs": runs,
    }


def write_ablation_outputs(
    report: dict[str, Any],
    output_dir: str | Path,
    *,
    stem: str = "retrieval_ablation",
) -> tuple[Path, Path]:
    """Write JSON and Markdown reports."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / f"{stem}.json"
    markdown_path = output_path / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_ablation_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def render_ablation_markdown(report: dict[str, Any]) -> str:
    """Render a Markdown table for the ablation report."""

    top_k = int(report["top_k"])
    lines = [
        "# Retrieval Ablation Report",
        "",
        f"- Questions: {report['question_count']}",
        f"- Base evidence count: {report['base_evidence_count']}",
        f"- Metric cutoff: @{top_k}",
        f"- Candidate pool: {report['candidate_top_k']}",
        "",
        "## Metrics",
        "",
        f"| Run | Evidence | Recall@{top_k} | Precision@{top_k} | MRR | NDCG@{top_k} | Hit/Total | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in report["runs"]:
        metrics = run["metrics"]
        notes = run["description"]
        if run["excluded_evidence_types"]:
            notes += " Excluded: " + ", ".join(run["excluded_evidence_types"]) + "."
        if run.get("temp_index"):
            notes += f" Temp index: `{run['temp_index']}`."
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(run["label"]),
                    str(run["evidence_count"]),
                    _fmt(metrics[f"recall_at_{top_k}"]),
                    _fmt(metrics[f"precision_at_{top_k}"]),
                    _fmt(metrics["mrr"]),
                    _fmt(metrics[f"ndcg_at_{top_k}"]),
                    f"{metrics['hit_count']}/{metrics['total']}",
                    _md(notes),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Configurations",
            "",
            "| Run | Search kwargs | Reranker | Excluded evidence types |",
            "|---|---|---|---|",
        ]
    )
    for run in report["runs"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(run["label"]),
                    "`" + json.dumps(run["search_kwargs"], ensure_ascii=False, sort_keys=True) + "`",
                    "yes" if run["use_reranker"] else "no",
                    _md(", ".join(run["excluded_evidence_types"]) or "none"),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _build_filtered_store(
    store: EvidenceStore,
    embedding_provider: EmbeddingProvider,
    *,
    excluded_types: tuple[str, ...],
    tmp_index_dir: Path,
) -> tuple[EvidenceStore, int]:
    evidence = [item for item in store.load_evidence() if item.evidence_type not in set(excluded_types)]
    suffix = "exclude_" + "_".join(excluded_types)
    filtered_store = store.create_empty(tmp_index_dir / suffix)
    filtered_store.build(evidence, embedding_provider)
    return filtered_store, len(evidence)


def _metrics(result: EvaluationResult, top_k: int) -> dict[str, Any]:
    return {
        "total": result.total,
        "hit_count": result.hit_count,
        f"recall_at_{top_k}": result.recall_at_k,
        f"precision_at_{top_k}": result.precision_at_k,
        "mrr": result.mrr,
        f"ndcg_at_{top_k}": result.ndcg_at_k,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
