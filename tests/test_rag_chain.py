"""测试RAG管道"""
import pytest
from unittest.mock import patch, MagicMock
from rag_chain import (
    format_context_from_docs,
    RAGChain,
)
from prompts import SYSTEM_PROMPT, QA_PROMPT_TEMPLATE


class FakeDocument:
    """模拟LangChain Document"""
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(content={self.page_content[:30]}...)"


def test_format_context_from_docs():
    """从检索文档格式化上下文文本"""
    docs = [
        FakeDocument(
            "肺磨玻璃结节（GGN）是指CT上表现为局灶性密度增高的病变。",
            metadata={"source": "放射诊断学.pdf", "chunk_index": 42}
        ),
        FakeDocument(
            "根据Fleischner指南，<6mm的纯磨玻璃结节无需随访。",
            metadata={"source": "Fleischner指南.pdf", "chunk_index": 15}
        ),
    ]

    context = format_context_from_docs(docs)

    assert "[来源1: 放射诊断学.pdf" in context
    assert "[来源2: Fleischner指南.pdf" in context
    assert "肺磨玻璃结节" in context
    assert "Fleischner指南" in context


def test_format_context_from_docs_empty():
    """空文档列表应返回提示信息"""
    context = format_context_from_docs([])

    assert "未检索到" in context or "无相关" in context


def test_build_prompt_with_context():
    """测试带上下文的prompt构建"""
    from rag_chain import build_prompt_with_context

    context = "测试参考内容"
    question = "什么是肺结节？"

    messages = build_prompt_with_context(context, question)

    assert len(messages) == 2  # system + user
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "测试参考内容" in messages[1]["content"]
    assert "什么是肺结节？" in messages[1]["content"]


def test_rag_chain_initialization():
    """RAGChain应正确初始化配置"""
    config = {
        "api_key": "test-key",
        "base_url": "https://test.api.com/v1",
        "model": "test-model",
    }
    chain = RAGChain(config)

    assert chain.config == config
    assert chain.api_key == "test-key"
    assert chain.base_url == "https://test.api.com/v1"
    assert chain.model == "test-model"


@patch("rag_chain.OpenAI")
def test_rag_chain_generate_calls_api(mock_openai_class):
    """生成方法应调用DeepSeek API"""
    # 设置mock
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "肺结节是指..."
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    config = {
        "api_key": "test-key",
        "base_url": "https://test.api.com/v1",
        "model": "test-model",
    }
    chain = RAGChain(config)
    answer, sources = chain.generate(
        question="什么是肺结节？",
        context_docs=[
            FakeDocument(
                "肺结节是直径≤3cm的局灶性密度增高区。",
                {"source": "guide.pdf", "chunk_index": 0}
            )
        ],
    )

    # 验证API被调用
    mock_client.chat.completions.create.assert_called_once()
    assert answer == "肺结节是指..."
    assert len(sources) == 1
    assert sources[0]["source"] == "guide.pdf"
