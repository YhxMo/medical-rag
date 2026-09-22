import pytest
from copy import deepcopy
from src.evaluation.resume_metrics import retrieval_metrics, choose_strategy, summarize
from scripts.generate_resume_v2_dataset import validate_splits
from scripts.prepare_resume_v2 import chapter_map


def test_ablation_context_ignores_score_roundoff_but_not_text_or_order():
    from src.evaluation.resume_reporting import answer_context

    source = {
        "id": "a",
        "source": "book",
        "page": 1,
        "type": "text",
        "content": "exact text",
        "scores": {"dense": 0.7482324122545101},
    }
    rounded = {**source, "scores": {"dense": 0.7482324122545102}}
    assert answer_context([source]) == answer_context([rounded])
    changed = {**source, "content": "other text"}
    assert answer_context([source]) != answer_context([changed])
    assert answer_context([source, changed]) != answer_context([changed, source])


def test_numeric_reference_answer_is_valid_but_blank_is_not():
    from scripts.generate_resume_v2_dataset import validate_tasks

    rows = [
        {
            "kind": kind,
            "question": "Read the displayed value?",
            "reference_answer": "99",
            "rationale": "Visible scale marking",
            "requires_visual": kind in ("chart", "screenshot"),
            "evidence_ids": ["e"],
            "expected_behavior": "refuse" if kind == "behavior" else "answer",
        }
        for kind in ("text", "chart", "screenshot", "behavior")
    ]
    assert len(validate_tasks({"tasks": rows}, {"e"}, "refuse")) == 4
    rows[0]["reference_answer"] = " "
    with pytest.raises(ValueError, match="Invalid task text"):
        validate_tasks({"tasks": rows}, {"e"}, "refuse")


def test_holdout_cannot_start_without_matching_development_review(
    tmp_path, monkeypatch
):
    import json
    import scripts.evaluate_resume_v2 as evaluator

    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]")
    with pytest.raises(RuntimeError, match="Development source review"):
        evaluator.online(None, dataset, "holdout")
    base = tmp_path / "artifacts/resume-v2"
    base.mkdir(parents=True)
    review = tmp_path / "answers-dev/review_completed.json"
    review.parent.mkdir()
    review.write_text("{}")
    (base / "dev_review_decision.json").write_text(
        json.dumps({"dataset_sha256": "wrong", "review_sha256": "wrong"})
    )
    with pytest.raises(ValueError, match="changed inputs"):
        evaluator.online(None, dataset, "holdout")


def test_online_report_separates_cache_latency_and_failed_denominators():
    from src.evaluation.resume_reporting import operational_metrics

    row = {
        "question_id": "q",
        "kind": "text",
        "mode": "auto",
        "status": "ok",
        "judge": {
            "correctness": 2,
            "completeness": 1,
            "groundedness": 2,
            "unsupported_claims": False,
        },
        "result": {
            "status": "answered",
            "sources": [{"id": "e", "content": "abc"}],
            "timings": {"total_seconds": 10},
            "image_paths": [],
            "calls": [
                {
                    "cache_hit": False,
                    "usage": {"input_tokens": 20, "output_tokens": 10},
                    "cost_upper_cny": 0.01,
                }
            ],
        },
    }
    cached = deepcopy(row)
    cached["result"]["calls"][0]["cache_hit"] = True
    cached["result"]["timings"]["total_seconds"] = 0.001
    failed = deepcopy(row)
    failed.update(status="failed", judge=None)
    failed["result"].update(status="service_error", calls=[], sources=[])
    report = operational_metrics(
        [row, cached, failed], [{"question_id": "q", "expected_evidence_ids": ["e"]}]
    )["text/auto"]
    assert report["source_hits"] == 2 and report["source_hit_denominator"] == 3
    assert report["failed"] == 1 and report["judged"] == 2
    assert report["latency_seconds_live_only"]["total_seconds"]["p50"] == 10
    assert report["reported_model_tokens"]["input_tokens"] == 20
    assert report["reported_answer_cost_cny"] == 0.01


def test_recall_is_not_hit_and_ndcg_has_cutoff():
    m = retrieval_metrics(["a", "b"], ["a", "x", "y", "z", "w", "b"])
    assert m["hit"] == 1 and m["recall"] == 0.5 and m["mrr"] == 1 and m["ndcg"] < 1


def test_failed_cases_stay_in_denominator():
    rows = [
        {
            "status": "ok",
            "seconds": 0.01,
            "context_chars": 5,
            "metrics": dict(hit=1, recall=1, mrr=1, ndcg=1),
        },
        {"status": "failed"},
    ]
    result = summarize(rows)
    assert result["hit"] == 0.5 and result["failed"] == 1


def test_selection_does_not_reward_latency_without_quality():
    base = dict(failed=0, scored=10, recall=0.9, ndcg=0.8, retrieval_p95_ms=50)
    assert (
        choose_strategy({"baseline": base, "chapter": {**base, "retrieval_p95_ms": 1}})
        == "baseline"
    )
    assert (
        choose_strategy({"baseline": base, "chapter": {**base, "recall": 1}})
        == "chapter"
    )


def test_image_hash_leakage_rejected():
    manifest = {
        "visual_pages": [
            {"group_id": "a", "split": "dev", "image_paths": ["p"]},
            {"group_id": "b", "split": "holdout", "image_paths": ["q"]},
        ],
        "image_registry": {"p": "same-image", "q": "same-image"},
    }
    rows = [
        {"group_id": "a", "split": "dev", "source_file": "one", "page": 1},
        {"group_id": "b", "split": "holdout", "source_file": "two", "page": 20},
    ]
    with pytest.raises(ValueError, match="Image leakage"):
        validate_splits(rows, manifest)


def test_neighbor_page_leakage_rejected():
    manifest = {
        "visual_pages": [
            {"group_id": "a", "split": "dev", "image_paths": ["p"]},
            {"group_id": "b", "split": "holdout", "image_paths": ["q"]},
        ],
        "image_registry": {"p": "different1", "q": "different2"},
    }
    rows = [
        {"group_id": "a", "split": "dev", "source_file": "book", "page": 1},
        {"group_id": "b", "split": "holdout", "source_file": "book", "page": 3},
    ]
    with pytest.raises(ValueError, match="Adjacent"):
        validate_splits(rows, manifest)


def test_chapter_unknown_and_bookmark_precedence():
    class Page:
        def get_text(self):
            return "ordinary body with no heading"

    class Doc(list):
        def get_toc(self):
            return []

    doc = Doc([Page(), Page()])
    assert chapter_map(doc)[1]["chapter_origin"] == "unknown"
    doc.get_toc = lambda: [[1, "Actual chapter", 2]]
    assert chapter_map(doc)[2]["chapter_path"] == ["Actual chapter"]
