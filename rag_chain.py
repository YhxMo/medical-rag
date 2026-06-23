"""RAG检索增强生成管道"""
from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI
from langchain_core.documents import Document

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    RETRIEVAL_TOP_K,
)
from prompts import (
    SYSTEM_PROMPT, QA_PROMPT_TEMPLATE,
    NO_CONTEXT_PROMPT_TEMPLATE, RAG_VS_NO_RAG_PROMPT,
)

logger = logging.getLogger(__name__)


def format_context_from_docs(docs: list[Document]) -> str:
    """将检索到的文档列表格式化为上下文文本。

    Args:
        docs: 检索到的LangChain Document列表

    Returns:
        格式化的上下文字符串，包含来源标注
    """
    if not docs:
        return "（未检索到相关参考资料）\n"

    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        parts.append(
            f"[来源{i}: {source}，第{chunk_idx}段]\n{doc.page_content}\n"
        )
    return "\n".join(parts)


def build_prompt_with_context(
    context: str,
    question: str,
) -> list[dict[str, str]]:
    """构建包含上下文的对话消息。

    Args:
        context: 格式化的参考上下文
        question: 用户问题

    Returns:
        OpenAI API 格式的消息列表
    """
    if "未检索到" in context or context.strip() == "":
        user_content = NO_CONTEXT_PROMPT_TEMPLATE.format(question=question)
    else:
        user_content = QA_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class RAGChain:
    """RAG检索增强生成链。

    整合检索与生成，提供端到端的问答接口。
    """

    def __init__(self, config: dict[str, str] | None = None):
        self.config = config or {
            "api_key": DEEPSEEK_API_KEY,
            "base_url": DEEPSEEK_BASE_URL,
            "model": DEEPSEEK_MODEL,
        }
        self.api_key = self.config["api_key"]
        self.base_url = self.config["base_url"]
        self.model = self.config["model"]

    def _get_client(self) -> OpenAI:
        """获取OpenAI兼容客户端（指向DeepSeek API）"""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate(
        self,
        question: str,
        context_docs: list[Document],
    ) -> tuple[str, list[dict[str, Any]]]:
        """基于检索到的文档生成回答。

        Args:
            question: 用户问题
            context_docs: 检索到的相关文档列表

        Returns:
            (answer_text, sources_list) 元组
            sources_list 每项包含 source, chunk_index, content 字段
        """
        context = format_context_from_docs(context_docs)
        messages = build_prompt_with_context(context, question)

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,  # 低温度以保证准确性
            max_tokens=1024,
        )

        answer = response.choices[0].message.content
        if answer is None:
            logger.warning("API returned None content (possible content filter/refusal)")
            answer = "（模型暂时无法生成回答，请稍后重试。）"

        # 提取引用来源
        sources = [
            {
                "source": doc.metadata.get("source", "未知"),
                "chunk_index": doc.metadata.get("chunk_index", -1),
                "content": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
            }
            for doc in context_docs
        ]

        logger.info("生成回答完成，长度=%d字符，引用%d个来源",
                    len(answer), len(sources))
        return answer, sources

    def generate_without_rag(self, question: str) -> str:
        """不使用RAG直接生成回答（用于对比评估）。

        Args:
            question: 用户问题

        Returns:
            模型回答文本
        """
        client = self._get_client()
        messages = [
            {"role": "system", "content": "你是一位专业的医学影像学助手。请提供专业准确的回答。\n\n[免责声明]\n以下内容仅供参考，不能替代专业医师的诊断和建议。如有健康疑虑，请及时就医。"},
            {"role": "user", "content": RAG_VS_NO_RAG_PROMPT.format(
                question=question
            )},
        ]
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content
        if answer is None:
            logger.warning("API returned None content (possible content filter/refusal)")
            answer = "（模型暂时无法生成回答，请稍后重试。）"
        return answer


def query_with_rag(
    question: str,
    vector_store,
    rag_chain: RAGChain,
    top_k: int = RETRIEVAL_TOP_K,
) -> dict[str, Any]:
    """完整的RAG问答流程：检索 + 生成。

    Args:
        question: 用户问题
        vector_store: ChromaDB向量存储
        rag_chain: RAGChain实例
        top_k: 检索返回的文档数

    Returns:
        包含 question, answer, sources, retrieved_count 的字典
    """
    # Step 1: 语义检索
    retrieved_docs = vector_store.similarity_search(question, k=top_k)
    logger.info("检索到 %d 篇相关文档", len(retrieved_docs))

    # Step 2: 增强生成
    answer, sources = rag_chain.generate(question, retrieved_docs)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_count": len(retrieved_docs),
    }
