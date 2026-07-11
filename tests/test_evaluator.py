from src.embedding.simple import HashEmbeddingProvider
from src.evaluation.ablation import RetrievalAblationSpec, render_ablation_markdown, run_retrieval_ablation
from src.evaluation.dataset import EvaluationQuestion
from src.evaluation.evaluator import evaluate_retrieval
from src.indexer.store import IndexStore
from src.schema import EvidenceItem


def test_evaluate_retrieval_reports_recall(tmp_path):
    evidence = [EvidenceItem("e1", "text", "book.pdf", "脑出血 CT 高密度影", 1, 1)]
    provider = HashEmbeddingProvider()
    store = IndexStore(tmp_path / "index")
    store.build(evidence, provider)

    result = evaluate_retrieval(
        [EvaluationQuestion("q1", "脑出血 CT", ("e1",), reviewed=True)],
        store,
        provider,
        top_k=1,
    )

    assert result.total == 1
    assert result.recall_at_k == 1.0


def test_evaluate_retrieval_reports_precision_mrr_ndcg_on_hit(tmp_path):
    evidence = [
        EvidenceItem("e1", "text", "book.pdf", "脑出血 CT 高密度影", 1, 1),
        EvidenceItem("e2", "text", "book.pdf", "肺炎 斑片状阴影", 2, 2),
    ]
    provider = HashEmbeddingProvider()
    store = IndexStore(tmp_path / "index")
    store.build(evidence, provider)

    result = evaluate_retrieval(
        [EvaluationQuestion("q1", "脑出血 CT", ("e1",), reviewed=True)],
        store,
        provider,
        top_k=1,
    )

    # Single expected id, single retrieved slot, and it is the correct one:
    # precision@1 = 1.0, reciprocal rank = 1.0 (rank 1 hit), ndcg@1 = 1.0.
    assert result.precision_at_k == 1.0
    assert result.mrr == 1.0
    assert result.ndcg_at_k == 1.0


def test_evaluate_retrieval_reports_zero_metrics_on_miss(tmp_path):
    evidence = [EvidenceItem("e1", "text", "book.pdf", "脑出血 CT 高密度影", 1, 1)]
    provider = HashEmbeddingProvider()
    store = IndexStore(tmp_path / "index")
    store.build(evidence, provider)

    result = evaluate_retrieval(
        [EvaluationQuestion("q1", "脑出血 CT", ("does-not-exist",), reviewed=True)],
        store,
        provider,
        top_k=1,
    )

    assert result.recall_at_k == 0.0
    assert result.precision_at_k == 0.0
    assert result.mrr == 0.0
    assert result.ndcg_at_k == 0.0


def test_evaluate_retrieval_applies_reranker(tmp_path):
    evidence = [
        EvidenceItem("e1", "text", "book.pdf", "脑出血 CT 高密度影", 1, 1),
        EvidenceItem("e2", "text", "book.pdf", "肺炎 斑片状阴影", 2, 2),
    ]
    provider = HashEmbeddingProvider()
    store = IndexStore(tmp_path / "index")
    store.build(evidence, provider)

    class ReverseReranker:
        """Test double: always promotes e2 to rank 1 regardless of retrieval order."""

        def rerank(self, query, hits, *, top_k=None):
            reordered = sorted(hits, key=lambda hit: hit.evidence.evidence_id, reverse=True)
            limited = reordered[:top_k] if top_k is not None else reordered
            return limited

    result = evaluate_retrieval(
        [EvaluationQuestion("q1", "脑出血 CT", ("e2",), reviewed=True)],
        store,
        provider,
        top_k=2,
        reranker=ReverseReranker(),
    )

    assert result.details[0]["retrieved_evidence_ids"][0] == "e2"
    assert result.recall_at_k == 1.0


def test_evaluate_retrieval_accepts_search_kwargs(tmp_path):
    evidence = [
        EvidenceItem("e1", "text", "book.pdf", "alpha target marker", 1, 1),
        EvidenceItem("e2", "text", "book.pdf", "gamma delta", 2, 2),
    ]
    provider = HashEmbeddingProvider()
    store = IndexStore(tmp_path / "index")
    store.build(evidence, provider)

    result = evaluate_retrieval(
        [EvaluationQuestion("q1", "target", ("e1",), reviewed=True)],
        store,
        provider,
        top_k=1,
        search_kwargs={"dense_top_k": 0, "bm25_top_k": 0, "final_top_k": 2},
    )

    assert result.recall_at_k == 0.0
    assert result.details[0]["retrieved_evidence_ids"] == []


def test_run_retrieval_ablation_reports_metrics_and_markdown(tmp_path):
    evidence = [
        EvidenceItem("e1", "text", "book.pdf", "alpha target marker", 1, 1),
        EvidenceItem("e2", "image_caption", "book.pdf", "gamma delta", 2, 2),
    ]
    provider = HashEmbeddingProvider()
    store = IndexStore(tmp_path / "index")
    store.build(evidence, provider)
    specs = [
        RetrievalAblationSpec(
            name="disabled",
            label="Disabled retrieval",
            description="No retrieval candidates.",
            search_kwargs={"dense_top_k": 0, "bm25_top_k": 0, "final_top_k": 2},
        )
    ]

    report = run_retrieval_ablation(
        [EvaluationQuestion("q1", "target", ("e1",), reviewed=True)],
        store,
        provider,
        top_k=1,
        candidate_top_k=2,
        specs=specs,
    )
    markdown = render_ablation_markdown(report)

    assert report["runs"][0]["metrics"]["recall_at_1"] == 0.0
    assert "| Disabled retrieval |" in markdown
