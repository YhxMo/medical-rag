"""测试肺结节随访工具化流程"""
from __future__ import annotations

import json

from langchain_core.documents import Document

from guideline_tools import (
    classify_nodule_type,
    create_followup_tools,
    generate_followup_recommendation,
    retrieve_guideline,
    run_followup_tool_flow,
)


class FakeVectorStore:
    """模拟向量库，记录检索参数并返回固定指南片段。"""

    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def similarity_search(self, query: str, k: int = 4):
        self.calls.append((query, k))
        return [
            Document(
                page_content="部分实性结节≥6mm且实性成分<6mm，建议3-6个月后复查低剂量CT。",
                metadata={"source": "Fleischner_2017.pdf", "chunk_index": 12},
            ),
            Document(
                page_content="随访建议应结合结节大小、密度类型和患者高危因素综合判断。",
                metadata={"source": "Chinese_consensus_2024.pdf", "chunk_index": 35},
            ),
        ][:k]


class FakeRAGChain:
    """模拟RAGChain，只提供OpenAI兼容客户端。"""

    def __init__(self, payload: dict):
        self.payload = payload
        self.messages = None

    def _get_client(self):
        outer = self

        class FakeCompletions:
            def create(self, **kwargs):
                outer.messages = kwargs["messages"]

                class Message:
                    content = json.dumps(outer.payload, ensure_ascii=False)

                class Choice:
                    message = Message()

                class Response:
                    choices = [Choice()]

                return Response()

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

        return FakeClient()


def test_retrieve_guideline_returns_context_and_numbered_sources():
    """retrieve_guideline应封装向量检索并返回可引用来源。"""
    vector_store = FakeVectorStore()

    result = retrieve_guideline(
        query="8mm部分实性肺结节随访建议",
        vector_store=vector_store,
        top_k=2,
    )

    assert vector_store.calls == [("8mm部分实性肺结节随访建议", 2)]
    assert result["retrieved_count"] == 2
    assert "部分实性结节" in result["context"]
    assert result["sources"][0]["source_id"] == 1
    assert result["sources"][0]["source"] == "Fleischner_2017.pdf"
    assert result["sources"][0]["chunk_index"] == 12


def test_classify_nodule_type_prefers_part_solid_over_ground_glass():
    """含实性成分的磨玻璃描述应归为部分实性结节。"""
    nodule_type = classify_nodule_type("右上肺8mm磨玻璃结节，内见约3mm实性成分")

    assert nodule_type == "part-solid"


def test_generate_followup_recommendation_structures_and_checks_citations():
    """结构化随访生成工具应解析JSON并补充引用自检。"""
    retrieval = retrieve_guideline(
        query="8mm部分实性肺结节随访建议",
        vector_store=FakeVectorStore(),
        top_k=2,
    )
    chain = FakeRAGChain(
        {
            "nodule_type": "part-solid",
            "recommendation": "建议3-6个月后复查低剂量胸部CT。[来源1: Fleischner_2017.pdf，第12段]",
            "followup_interval": "3-6个月",
            "risk_level": "需短期随访",
            "rationale": "结节为部分实性且直径≥6mm。[来源1: Fleischner_2017.pdf，第12段]",
            "evidence": [{"source_id": 1, "summary": "部分实性结节≥6mm建议3-6个月随访"}],
            "disclaimer": "仅供参考，不能替代专业医师诊断。",
        }
    )

    result = generate_followup_recommendation(
        case_description="右上肺8mm部分实性结节，实性成分3mm",
        guideline_result=retrieval,
        rag_chain=chain,
        nodule_type="part-solid",
    )

    assert result["nodule_type"] == "part-solid"
    assert result["followup_interval"] == "3-6个月"
    assert result["citation_check"]["status"] == "pass"
    assert result["citation_check"]["cited_source_ids"] == [1]
    assert "部分实性" in chain.messages[-1]["content"]


def test_run_followup_tool_flow_runs_retrieve_classify_generate_and_self_check():
    """编排函数应完成检索、结节类型判断、建议生成和引用自检。"""
    chain = FakeRAGChain(
        {
            "nodule_type": "part-solid",
            "recommendation": "建议3-6个月后复查低剂量胸部CT。[来源1: Fleischner_2017.pdf，第12段]",
            "followup_interval": "3-6个月",
            "risk_level": "需短期随访",
            "rationale": "依据部分实性结节随访建议。[来源1: Fleischner_2017.pdf，第12段]",
            "evidence": [{"source_id": 1, "summary": "部分实性结节≥6mm建议3-6个月随访"}],
            "disclaimer": "仅供参考，不能替代专业医师诊断。",
        }
    )

    result = run_followup_tool_flow(
        case_description="右上肺8mm磨玻璃结节，内见3mm实性成分",
        vector_store=FakeVectorStore(),
        rag_chain=chain,
        top_k=2,
    )

    assert result["nodule_type"] == "part-solid"
    assert result["recommendation"]["citation_check"]["status"] == "pass"
    assert [step["step"] for step in result["tool_trace"]] == [
        "retrieve_guideline",
        "classify_nodule_type",
        "generate_followup_recommendation",
        "self_check_citations",
    ]


def test_create_followup_tools_returns_langchain_tool_objects():
    """随访流程应暴露LangChain @tool工具对象，便于agent绑定。"""
    chain = FakeRAGChain(
        {
            "nodule_type": "part-solid",
            "recommendation": "建议3-6个月后复查低剂量胸部CT。[来源1: Fleischner_2017.pdf，第12段]",
            "followup_interval": "3-6个月",
            "risk_level": "需短期随访",
            "rationale": "依据部分实性结节随访建议。[来源1: Fleischner_2017.pdf，第12段]",
            "evidence": [{"source_id": 1, "summary": "部分实性结节≥6mm建议3-6个月随访"}],
            "disclaimer": "仅供参考，不能替代专业医师诊断。",
        }
    )

    tools = create_followup_tools(FakeVectorStore(), chain, top_k=2)
    tools_by_name = {tool.name: tool for tool in tools}

    assert set(tools_by_name) == {
        "retrieve_guideline",
        "generate_followup_recommendation",
    }
    assert all(hasattr(tool, "invoke") for tool in tools)

    retrieval = tools_by_name["retrieve_guideline"].invoke(
        {"query": "8mm部分实性肺结节随访建议"}
    )
    recommendation = tools_by_name["generate_followup_recommendation"].invoke(
        {
            "case_description": "右上肺8mm部分实性结节，实性成分3mm",
            "nodule_type": "part-solid",
            "guideline_result": retrieval,
        }
    )

    assert retrieval["retrieved_count"] == 2
    assert recommendation["citation_check"]["status"] == "pass"
