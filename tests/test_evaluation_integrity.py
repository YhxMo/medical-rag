"""Synthetic-only regression tests for metric definitions and report privacy."""
import json
import math

import pytest

from src.embedding.simple import HashEmbeddingProvider
from src.evaluation.ablation import RetrievalAblationSpec, run_retrieval_ablation, write_ablation_outputs
from src.evaluation.dataset import EvaluationQuestion, load_dataset
from src.evaluation.evaluator import evaluate_retrieval, save_result
from src.schema import EvidenceItem, RetrievalHit


class RankedStore:
    def __init__(self, ids):
        self.ids = ids

    def load_evidence(self):
        return [EvidenceItem(eid, "text", "synthetic", "fixture") for eid in dict.fromkeys(self.ids)]

    def search(self, query, provider, **kwargs):
        return [RetrievalHit(EvidenceItem(eid, "text", "synthetic", "fixture"), 1.0, i)
                for i, eid in enumerate(self.ids, 1)]


def evaluate(ids, expected=("a", "b"), k=5, **kwargs):
    return evaluate_retrieval(
        [EvaluationQuestion("private-id", "SYNTHETIC_PRIVATE_QUESTION", expected, reviewed=True)],
        RankedStore(ids), HashEmbeddingProvider(), top_k=k, **kwargs,
    )


def test_partial_recall_differs_from_hit_rate_and_short_results_are_penalized():
    result = evaluate(["a"])
    assert result.recall_at_k == 0.5
    assert result.hit_rate_at_k == 1.0
    assert result.precision_at_k == 0.2
    assert result.ndcg_at_k == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_duplicate_evidence_cannot_inflate_metrics():
    result = evaluate(["a", "a", "b"], k=2)
    assert result.recall_at_k == result.precision_at_k == result.ndcg_at_k == 1.0
    assert result.details[0]["retrieved_evidence_ids"] == ["a", "b"]


def test_cutoff_applies_even_if_reranker_returns_too_many_hits():
    class UnboundedReranker:
        def rerank(self, query, hits, **kwargs):
            return hits
    result = evaluate(["x", "y", "a"], k=2, reranker=UnboundedReranker())
    assert result.recall_at_k == result.mrr == result.ndcg_at_k == 0.0


def test_macro_recall_and_mrr_for_multiple_questions():
    result = evaluate_retrieval(
        [EvaluationQuestion("1", "one", ("a", "b")), EvaluationQuestion("2", "two", ("z",))],
        RankedStore(["x", "a"]), HashEmbeddingProvider(), top_k=2,
    )
    assert result.recall_at_k == 0.25
    assert result.hit_rate_at_k == 0.5
    assert result.precision_at_k == result.mrr == 0.25


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_cutoff_is_rejected(k):
    with pytest.raises(ValueError):
        evaluate(["a"], k=k)


def test_empty_or_unlabelled_evaluation_fails():
    with pytest.raises(ValueError):
        evaluate_retrieval([], RankedStore([]), HashEmbeddingProvider())
    with pytest.raises(ValueError):
        evaluate(["a"], expected=())


def test_saved_summary_never_contains_question_ids_or_raw_content(tmp_path):
    result = evaluate(["PRIVATE_EVIDENCE_ID"], expected=("PRIVATE_EVIDENCE_ID",))
    result.details[0]["api_key"] = "SYNTHETIC_PRIVATE_SECRET"
    path = tmp_path / "result.json"
    save_result(path, result)
    raw = path.read_text()
    assert "PRIVATE" not in raw
    assert "private-id" not in raw
    assert "details" not in raw
    assert json.loads(raw)["metrics_version"] == "retrieval-v2"


def test_ablation_persistence_strips_raw_details_and_custom_labels(tmp_path):
    report = run_retrieval_ablation(
        [EvaluationQuestion("SYNTHETIC_PRIVATE_ID", "SYNTHETIC_PRIVATE_QUESTION", ("a",))],
        RankedStore(["a"]), HashEmbeddingProvider(), top_k=1, candidate_top_k=2,
        specs=[RetrievalAblationSpec("PRIVATE", "PRIVATE", "PRIVATE", {"final_top_k": 2})],
    )
    report["runs"][0]["details"] = [{"question": "PRIVATE"}]
    report["runs"][0]["temp_index"] = "/PRIVATE/index"
    paths = write_ablation_outputs(report, tmp_path)
    assert all("PRIVATE" not in path.read_text() for path in paths)
    assert json.loads(paths[0].read_text())["runs"][0]["label"] == "Custom run 1"


@pytest.mark.parametrize("items", [
    [],
    [{"question_id": "q", "question": "fixture", "reviewed": "false"}],
    [{"question_id": "q", "question": "fixture", "reviewed": True}],
    [{"question_id": "q", "question": "fixture", "reviewed": True, "expected_evidence_ids": "a"}],
    [{"question_id": "q", "question": "fixture", "reviewed": True, "expected_evidence_ids": ["a", "a"]}],
    [{"question_id": "q", "question": "fixture", "reviewed": True, "expected_evidence_ids": ["a"]}] * 2,
])
def test_invalid_dataset_fails_without_echoing_content(tmp_path, items):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(items))
    with pytest.raises(ValueError) as exc:
        load_dataset(path)
    assert "fixture" not in str(exc.value)


def test_missing_dataset_does_not_silently_produce_zero_metrics(tmp_path):
    with pytest.raises(ValueError):
        load_dataset(tmp_path / "missing.json")


def test_cli_passes_full_candidate_pool_to_reranker_and_saves_safe_report(tmp_path, monkeypatch):
    import src.cli as cli
    dataset = tmp_path / "questions.json"
    dataset.write_text(json.dumps([{
        "question_id": "PRIVATE_ID", "question": "PRIVATE_TEXT", "reviewed": True,
        "expected_evidence_ids": ["b"],
    }]))
    config = tmp_path / "config.yaml"
    config.write_text('evaluation:\n  test_dataset: ' + json.dumps(str(dataset)) + '\n')

    class CandidateStore(RankedStore):
        def search(self, query, provider, **kwargs):
            assert kwargs == {"dense_top_k": 4, "bm25_top_k": 4, "final_top_k": 4}
            return super().search(query, provider, **kwargs)

    class PromoteB:
        def rerank(self, query, hits, **kwargs):
            assert len(hits) == 4
            return sorted(hits, key=lambda hit: hit.evidence.evidence_id != "b")

    monkeypatch.setattr(cli, "create_index_store", lambda settings: CandidateStore(["x", "y", "z", "b"]))
    monkeypatch.setattr(cli, "_build_reranker", lambda *args: PromoteB())
    output = tmp_path / "result.json"
    assert cli.main(["evaluate", "--config", str(config), "--embedding", "hash", "--reranker", "none",
                     "--top-k", "1", "--candidate-top-k", "4", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["recall_at_k"] == 1.0
    assert report["search_options"]["final_top_k"] == 4
    assert "PRIVATE" not in output.read_text()
