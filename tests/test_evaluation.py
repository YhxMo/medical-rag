"""Synthetic regression tests for metric definitions and experiment records."""

import json
import math

import pytest

from src.embedding.simple import HashEmbeddingProvider
from src.evaluation.dataset import EvaluationQuestion, load_dataset
from src.evaluation.evaluator import evaluate_retrieval, save_result
from src.schema import EvidenceItem, RetrievalHit


class RankedStore:
    def __init__(self, ids):
        self.ids = ids

    def load_evidence(self):
        return [
            EvidenceItem(eid, "text", "synthetic", "fixture") for eid in dict.fromkeys(self.ids)
        ]

    def search(self, query, provider, **kwargs):
        return [
            RetrievalHit(EvidenceItem(eid, "text", "synthetic", "fixture"), 1.0, i)
            for i, eid in enumerate(self.ids, 1)
        ]


def evaluate(ids, expected=("a", "b"), k=5, **kwargs):
    return evaluate_retrieval(
        [EvaluationQuestion("private-id", "SYNTHETIC_PRIVATE_QUESTION", expected, reviewed=True)],
        RankedStore(ids),
        HashEmbeddingProvider(),
        top_k=k,
        **kwargs,
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
        RankedStore(["x", "a"]),
        HashEmbeddingProvider(),
        top_k=2,
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


def test_saved_report_preserves_per_question_records_and_refuses_overwrite(tmp_path):
    result = evaluate(["a"], expected=("a",))
    path = tmp_path / "run.json"
    save_result(path, result, inputs={"dataset_sha256": "fixture"})
    raw = path.read_text()
    report = json.loads(raw)
    assert report["metrics"]["recall_at_k"] == 1
    assert report["details"][0]["retrieved_evidence_ids"] == ["a"]
    assert report["inputs"]["dataset_sha256"] == "fixture"
    with pytest.raises(FileExistsError):
        save_result(path, result)
    assert path.read_text() == raw


@pytest.mark.parametrize(
    "items",
    [
        [],
        [{"question_id": "q", "question": "fixture", "reviewed": "false"}],
        [{"question_id": "q", "question": "fixture", "reviewed": True}],
        [
            {
                "question_id": "q",
                "question": "fixture",
                "reviewed": True,
                "expected_evidence_ids": "a",
            }
        ],
        [
            {
                "question_id": "q",
                "question": "fixture",
                "reviewed": True,
                "expected_evidence_ids": ["a", "a"],
            }
        ],
        [
            {
                "question_id": "q",
                "question": "fixture",
                "reviewed": True,
                "expected_evidence_ids": ["a"],
            }
        ]
        * 2,
    ],
)
def test_invalid_dataset_fails_without_echoing_content(tmp_path, items):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(items))
    with pytest.raises(ValueError) as exc:
        load_dataset(path)
    assert "fixture" not in str(exc.value)


def test_missing_dataset_does_not_silently_produce_zero_metrics(tmp_path):
    with pytest.raises(ValueError):
        load_dataset(tmp_path / "missing.json")
