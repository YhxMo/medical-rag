"""端到端集成测试"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from config import ROOT, DATA_DIR, VECTOR_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from ingest import split_text_into_chunks
from rag_chain import RAGChain, query_with_rag, format_context_from_docs
from evaluate import compute_source_coverage, score_answer_relevance
from prompts import SYSTEM_PROMPT, QA_PROMPT_TEMPLATE, NO_CONTEXT_PROMPT_TEMPLATE


MEDICAL_SAMPLE = """
第一章 肺结节影像学

1.1 概述
肺结节是指肺内直径≤3cm的类圆形或不规则形病灶，影像学表现为密度增高影。
根据密度特征，肺结节可分为实性结节、部分实性结节和纯磨玻璃结节。

1.2 磨玻璃结节（GGN）
磨玻璃结节是指CT上表现为局灶性密度增高但不掩盖其内血管和支气管走行的病变。
纯磨玻璃结节（pGGN）不含实性成分，部分实性结节（PSN）则含有不同比例的实性成分。

1.3 Fleischner指南建议
根据Fleischner学会2017年指南：
- <6mm的纯磨玻璃结节无需常规随访
- ≥6mm的纯磨玻璃结节建议6-12个月后CT随访
- 部分实性结节≥6mm且实性成分<6mm建议3-6个月随访
"""


class FakeVectorStore:
    """模拟ChromaDB向量存储"""
    def similarity_search(self, query, k=4):
        chunks = split_text_into_chunks(MEDICAL_SAMPLE, chunk_size=300, chunk_overlap=50)
        from langchain_core.documents import Document
        return [
            Document(page_content=c, metadata={"source": "test.pdf", "chunk_index": i})
            for i, c in enumerate(chunks[:k])
        ]


@patch("rag_chain.OpenAI")
def test_full_rag_pipeline_end_to_end(mock_openai):
    """测试完整RAG管道：问题→检索→生成→答案"""
    # Mock DeepSeek API响应
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = (
        "肺磨玻璃结节（GGN）是CT上表现为局灶性密度增高但不掩盖血管走行的病变。"
        "根据Fleischner指南，<6mm的纯GGN无需常规随访。[来源1: test.pdf]"
    )
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    # 初始化
    vector_store = FakeVectorStore()
    rag_chain = RAGChain({
        "api_key": "test-key",
        "base_url": "https://test.com/v1",
        "model": "test-model",
    })

    # 执行RAG问答
    result = query_with_rag("什么是肺磨玻璃结节？", vector_store, rag_chain)

    # 验证结果结构
    assert "question" in result
    assert "answer" in result
    assert "sources" in result
    assert result["answer"] is not None
    assert len(result["sources"]) > 0

    # 验证来源信息
    for src in result["sources"]:
        assert "source" in src
        assert "content" in src


def test_end_to_end_no_api_calls():
    """验证各组件可独立运行（无需API）"""
    # 1. 分块功能
    chunks = split_text_into_chunks(MEDICAL_SAMPLE, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 1
    assert any("肺结节" in c or "GGN" in c or "Fleischner" in c for c in chunks if len(c) > 50)

    # 2. 上下文格式化
    from langchain_core.documents import Document
    docs = [
        Document(page_content=chunks[0], metadata={"source": "test.pdf", "chunk_index": 0})
    ]
    context = format_context_from_docs(docs)
    assert "[来源1: test.pdf" in context

    # 3. Prompt构建
    from rag_chain import build_prompt_with_context
    messages = build_prompt_with_context(context, "测试问题？")
    assert messages[0]["role"] == "system"
    assert "专业" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "测试问题" in messages[1]["content"]

    # 4. 评估指标
    coverage = compute_source_coverage(
        "根据[来源1: guide.pdf]的指南，肺结节需要随访。",
        [{"source": "guide.pdf", "chunk_index": 0}]
    )
    assert coverage == 1.0

    relevance = score_answer_relevance(
        "肺磨玻璃结节是CT上的密度增高影。",
        ["肺磨玻璃结节", "CT", "密度增高"]
    )
    assert relevance > 0.5


def test_config_creates_directories():
    """config模块应自动创建必要的目录"""
    assert DATA_DIR.exists()
    assert VECTOR_DB_DIR.exists()


def test_requirements_txt_exists():
    """requirements.txt应存在且包含必要依赖"""
    req_path = ROOT / "requirements.txt"
    assert req_path.exists()

    content = req_path.read_text(encoding="utf-8")
    required = ["langchain", "chromadb", "sentence-transformers", "gradio", "openai"]
    for dep in required:
        assert dep in content, f"缺少依赖: {dep}"


def test_system_prompt_contains_safety():
    """系统Prompt必须包含免责声明和安全规则"""
    assert "参考资料" in SYSTEM_PROMPT
    assert "免责声明" in SYSTEM_PROMPT
    assert "专业医师" in SYSTEM_PROMPT
