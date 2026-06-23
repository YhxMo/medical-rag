"""RAG问答系统评估"""
from __future__ import annotations

import json
import logging
from typing import Any

from config import EVAL_RESULTS_FILE, EVAL_QUESTIONS_FILE

logger = logging.getLogger(__name__)

# 默认评估问题集
DEFAULT_EVAL_QUESTIONS = [
    "什么是肺磨玻璃结节（GGN）？",
    "肺结节如何根据Fleischner指南进行随访管理？",
    "胸部CT的肺窗和纵隔窗有什么区别？",
    "部分实性结节和纯磨玻璃结节有什么不同？",
    "肺部影像学检查中，哪些特征提示恶性可能？",
]

# 每个问题的关键词映射，用于准确的相关性评估
DEFAULT_QUESTION_KEYWORDS = {
    "什么是肺磨玻璃结节（GGN）？": ["磨玻璃结节", "GGN", "CT", "密度增高", "局灶性"],
    "肺结节如何根据Fleischner指南进行随访管理？": ["Fleischner", "随访", "结节大小", "风险分层", "指南"],
    "胸部CT的肺窗和纵隔窗有什么区别？": ["肺窗", "纵隔窗", "肺实质", "纵隔结构", "软组织"],
    "部分实性结节和纯磨玻璃结节有什么不同？": ["部分实性", "纯磨玻璃", "实性成分", "密度", "鉴别"],
    "肺部影像学检查中，哪些特征提示恶性可能？": ["恶性", "毛刺征", "分叶征", "胸膜凹陷", "生长速度"],
}


def extract_keywords(question: str) -> list[str]:
    """从问题中提取关键词用于相关性评分。

    Args:
        question: 评估问题文本

    Returns:
        关键词列表
    """
    # 优先检查预设映射
    for q_pattern, keywords in DEFAULT_QUESTION_KEYWORDS.items():
        if q_pattern in question or question in q_pattern:
            return keywords
    # 回退：尝试jieba分词
    try:
        import jieba
        words = list(jieba.cut(question))
        keywords = [w.strip() for w in words
                    if len(w.strip()) >= 2 and w.strip() not in "？，。！？的是什么哪些如何"]
        if keywords:
            return keywords[:10]
    except ImportError:
        pass
    # 最终回退：字符二元组
    clean = question.replace("？", "").replace("，", "").replace("（", "").replace("）", "").replace("、", "")
    if len(clean) >= 2:
        return [clean[i:i+2] for i in range(0, len(clean)-1, max(1, len(clean)//10))][:10]
    return [clean]


def compute_source_coverage(
    answer: str,
    sources: list[dict[str, Any]],
) -> float:
    """计算回答中引用来源的覆盖率。

    通过检测回答中是否引用了[来源N]标记来判断。

    Args:
        answer: 模型生成的回答文本
        sources: 检索到的来源列表

    Returns:
        覆盖率 (0.0 ~ 1.0)，表示被引用的来源比例
    """
    if not sources:
        return 0.0

    cited_count = 0
    for i in range(1, len(sources) + 1):
        marker_bracketed = f"[来源{i}:"
        marker_bare = f"来源{i}"
        if marker_bracketed in answer or marker_bare in answer:
            cited_count += 1

    return cited_count / len(sources)


def score_answer_relevance(
    answer: str,
    expected_keywords: list[str],
) -> float:
    """基于关键词匹配的自动相关性评分。

    Args:
        answer: 模型回答
        expected_keywords: 预期出现的关键词列表

    Returns:
        相关性分数 (0.0 ~ 1.0)
    """
    if not expected_keywords:
        return 1.0

    matched = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
    return matched / len(expected_keywords)


def run_comparative_eval(
    vector_store,
    rag_chain,
    questions: list[str] | None = None,
    output_file: str | None = None,
) -> dict[str, Any]:
    """运行RAG vs 纯LLM对比评估。

    对每个问题分别用RAG和纯LLM回答，计算自动化指标。

    Args:
        vector_store: ChromaDB向量存储
        rag_chain: RAGChain实例
        questions: 评估问题列表，None则使用默认问题集
        output_file: 结果输出JSON路径

    Returns:
        评估结果字典
    """
    from rag_chain import query_with_rag

    if questions is None:
        # 尝试从文件加载，否则使用默认问题集
        if EVAL_QUESTIONS_FILE.exists():
            with open(EVAL_QUESTIONS_FILE, "r", encoding="utf-8") as f:
                questions = json.load(f)
        else:
            questions = DEFAULT_EVAL_QUESTIONS

    results: list[dict[str, Any]] = []
    total_coverage = 0.0
    total_relevance = 0.0

    for question in questions:
        logger.info("评估问题: %s", question[:50])

        # RAG 回答
        rag_result = query_with_rag(question, vector_store, rag_chain)
        rag_answer = rag_result["answer"]

        # 无RAG回答
        no_rag_answer = rag_chain.generate_without_rag(question)

        # 计算指标
        coverage = compute_source_coverage(rag_answer, rag_result["sources"])

        # 使用关键词映射提取有意义的关键词用于相关性评分
        keywords = extract_keywords(question)
        relevance = score_answer_relevance(rag_answer, keywords)

        total_coverage += coverage
        total_relevance += relevance

        results.append({
            "question": question,
            "rag_answer": rag_answer,
            "no_rag_answer": no_rag_answer,
            "rag_sources": rag_result["sources"],
            "source_coverage": round(coverage, 3),
            "relevance_score": round(relevance, 3),
        })

    n = len(questions)
    eval_results = {
        "questions": results,
        "summary": {
            "total_questions": n,
            "avg_source_coverage": round(total_coverage / n, 3) if n else 0,
            "avg_relevance_score": round(total_relevance / n, 3) if n else 0,
        },
    }

    output_file = output_file or str(EVAL_RESULTS_FILE)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)

    logger.info("评估完成，结果已保存到 %s", output_file)
    return eval_results


def print_eval_summary(eval_results: dict[str, Any]) -> None:
    """打印评估摘要到控制台。"""
    summary = eval_results["summary"]
    print("\n" + "=" * 50)
    print("RAG 系统评估结果")
    print("=" * 50)
    print(f"评估问题数:     {summary['total_questions']}")
    print(f"平均来源覆盖率: {summary['avg_source_coverage']:.1%}")
    print(f"平均关键词得分: {summary['avg_relevance_score']:.1%}")
    print("=" * 50)

    for i, q in enumerate(eval_results["questions"], 1):
        print(f"\n问题{i}: {q['question']}")
        print(f"  来源覆盖率: {q['source_coverage']:.1%}")
        print(f"  关键词得分: {q['relevance_score']:.1%}")
