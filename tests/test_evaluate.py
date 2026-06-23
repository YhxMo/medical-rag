"""测试评估指标"""
import pytest
from evaluate import (
    compute_source_coverage,
    score_answer_relevance,
    run_comparative_eval,
)


def test_compute_source_coverage_full():
    """回答中引用了所有来源"""
    answer = "根据[来源1]和[来源2]，肺结节的管理需参照Fleischner指南。"
    sources = [
        {"source": "guide1.pdf", "chunk_index": 3},
        {"source": "guide2.pdf", "chunk_index": 7},
    ]

    coverage = compute_source_coverage(answer, sources)

    assert 0.0 <= coverage <= 1.0
    # 引用了来源1和来源2 - 覆盖率应为1.0
    assert coverage == 1.0


def test_compute_source_coverage_none():
    """回答中未引用任何来源引用标记"""
    answer = "根据相关指南，肺结节需要随访。"
    sources = [
        {"source": "guide1.pdf", "chunk_index": 3},
        {"source": "guide2.pdf", "chunk_index": 7},
    ]

    coverage = compute_source_coverage(answer, sources)

    assert coverage == 0.0


def test_compute_source_coverage_empty_sources():
    """无来源时覆盖率为0"""
    coverage = compute_source_coverage("一些回答", [])

    assert coverage == 0.0


def test_score_answer_relevance_contains_keywords():
    """回答包含预期关键词应得高分"""
    answer = "肺磨玻璃结节（GGN）是CT上表现为局灶性密度增高但不掩盖血管走行的病变。"
    expected_keywords = ["肺磨玻璃结节", "GGN", "CT", "密度增高"]

    score = score_answer_relevance(answer, expected_keywords)

    assert 0.0 <= score <= 1.0
    assert score > 0.5  # 至少命中一半关键词


def test_score_answer_relevance_no_match():
    """回答不包含任何关键词应得0分"""
    answer = "这个问题与肺部无关。"
    expected_keywords = ["肺磨玻璃结节", "GGN", "Fleischner"]

    score = score_answer_relevance(answer, expected_keywords)

    assert score == 0.0


def test_run_comparative_eval_structure(tmp_path):
    """对比评估应输出正确结构的JSON"""
    import json

    results = {
        "questions": [
            {
                "question": "测试问题1",
                "rag_answer": "带引用的回答 [来源：doc.pdf]",
                "no_rag_answer": "纯模型回答",
                "rag_sources": [{"source": "doc.pdf", "chunk_index": 0}],
                "source_coverage": 1.0,
                "relevance_score": 0.8,
            }
        ],
        "summary": {
            "total_questions": 1,
            "avg_source_coverage": 1.0,
            "avg_relevance_score": 0.8,
        },
    }

    result_file = tmp_path / "eval_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(result_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert len(loaded["questions"]) == 1
    assert loaded["summary"]["total_questions"] == 1
    assert loaded["summary"]["avg_source_coverage"] == 1.0
