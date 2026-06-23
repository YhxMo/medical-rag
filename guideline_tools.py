"""肺结节指南检索与随访建议工具。"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.documents import Document
from langchain.tools import tool

from config import DEEPSEEK_MODEL, RETRIEVAL_TOP_K
from prompts import FOLLOWUP_PROMPT_TEMPLATE, FOLLOWUP_SYSTEM_PROMPT
from rag_chain import format_context_from_docs

logger = logging.getLogger(__name__)


RETRIEVE_GUIDELINE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve_guideline",
        "description": "检索肺结节随访相关指南片段，返回带编号的可引用来源。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "面向指南知识库的检索问题，例如'8mm部分实性肺结节随访建议'。",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的指南片段数量。",
                    "minimum": 1,
                    "maximum": 10,
                    "default": RETRIEVAL_TOP_K,
                },
            },
            "required": ["query"],
        },
    },
}


GENERATE_FOLLOWUP_RECOMMENDATION_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "generate_followup_recommendation",
        "description": "根据病例描述、结节类型和检索证据生成结构化肺结节随访建议。",
        "parameters": {
            "type": "object",
            "properties": {
                "case_description": {
                    "type": "string",
                    "description": "肺结节影像描述和必要临床风险信息。",
                },
                "nodule_type": {
                    "type": "string",
                    "enum": [
                        "solid",
                        "pure-ground-glass",
                        "part-solid",
                        "multiple",
                        "unknown",
                    ],
                    "description": "根据病例描述判断的结节类型。",
                },
            },
            "required": ["case_description", "nodule_type"],
        },
    },
}


def get_followup_tool_schemas() -> list[dict[str, Any]]:
    """返回OpenAI兼容的工具定义。"""
    return [
        RETRIEVE_GUIDELINE_TOOL_SCHEMA,
        GENERATE_FOLLOWUP_RECOMMENDATION_TOOL_SCHEMA,
    ]


def create_followup_tools(
    vector_store,
    rag_chain,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[Any]:
    """创建LangChain 1.x可绑定的肺结节随访工具。"""

    @tool("retrieve_guideline")
    def retrieve_guideline_tool(query: str) -> dict[str, Any]:
        """检索肺结节随访相关指南片段，返回带编号的可引用来源。"""
        return retrieve_guideline(
            query=query,
            vector_store=vector_store,
            top_k=top_k,
        )

    @tool("generate_followup_recommendation")
    def generate_followup_recommendation_tool(
        case_description: str,
        nodule_type: str,
        guideline_result: dict[str, Any],
    ) -> dict[str, Any]:
        """根据病例描述、结节类型和检索证据生成结构化肺结节随访建议。"""
        return generate_followup_recommendation(
            case_description=case_description,
            guideline_result=guideline_result,
            rag_chain=rag_chain,
            nodule_type=nodule_type,
        )

    return [
        retrieve_guideline_tool,
        generate_followup_recommendation_tool,
    ]


def _doc_to_source(doc: Document, source_id: int) -> dict[str, Any]:
    content = doc.page_content
    return {
        "source_id": source_id,
        "source": doc.metadata.get("source", "未知"),
        "chunk_index": doc.metadata.get("chunk_index", -1),
        "content": content[:300] + ("..." if len(content) > 300 else ""),
    }


def retrieve_guideline(
    query: str,
    vector_store,
    top_k: int = RETRIEVAL_TOP_K,
) -> dict[str, Any]:
    """封装指南检索工具，返回上下文与编号来源。"""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("检索问题不能为空")
    if top_k < 1:
        raise ValueError("top_k必须大于等于1")

    docs = vector_store.similarity_search(normalized_query, k=top_k)
    sources = [_doc_to_source(doc, i) for i, doc in enumerate(docs, start=1)]
    context = format_context_from_docs(docs)
    logger.info("retrieve_guideline检索到%d条指南片段", len(docs))

    return {
        "query": normalized_query,
        "retrieved_count": len(docs),
        "context": context,
        "sources": sources,
    }


def classify_nodule_type(case_description: str) -> str:
    """从病例描述中粗略判断结节类型。"""
    text = case_description.lower()

    part_solid_markers = [
        "部分实性",
        "半实性",
        "混合磨玻璃",
        "实性成分",
        "part-solid",
        "part solid",
        "mixed ground-glass",
    ]
    if any(marker in text for marker in part_solid_markers):
        return "part-solid"

    if "多发" in text or "multiple" in text:
        return "multiple"

    pure_ground_glass_markers = [
        "纯磨玻璃",
        "纯ggn",
        "pggn",
        "pure ground-glass",
        "pure ground glass",
    ]
    if any(marker in text for marker in pure_ground_glass_markers):
        return "pure-ground-glass"

    if ("磨玻璃" in text or "ground-glass" in text or "ground glass" in text) and "实性" not in text:
        return "pure-ground-glass"

    solid_markers = ["实性结节", "solid nodule", "solid"]
    if any(marker in text for marker in solid_markers):
        return "solid"

    return "unknown"


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return stripped[start:end + 1]
    return stripped


def _parse_structured_json(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(_strip_json_fence(content))
    except json.JSONDecodeError:
        logger.warning("结构化随访建议JSON解析失败，使用原始文本回退")
        return {
            "nodule_type": "unknown",
            "recommendation": content.strip(),
            "followup_interval": "证据不足",
            "risk_level": "证据不足",
            "rationale": "模型未返回合法JSON，无法完成结构化解析。",
            "evidence": [],
            "disclaimer": "以下内容仅供参考，不能替代专业医师的诊断和建议。",
        }

    if not isinstance(parsed, dict):
        return {
            "nodule_type": "unknown",
            "recommendation": str(parsed),
            "followup_interval": "证据不足",
            "risk_level": "证据不足",
            "rationale": "模型返回的JSON不是对象。",
            "evidence": [],
            "disclaimer": "以下内容仅供参考，不能替代专业医师的诊断和建议。",
        }
    return parsed


def self_check_citations(
    structured_result: dict[str, Any],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """检查结构化建议是否引用了检索来源。"""
    valid_source_ids = {source["source_id"] for source in sources}
    serialized = json.dumps(structured_result, ensure_ascii=False)
    cited_source_ids = {int(match) for match in re.findall(r"\[来源(\d+)", serialized)}

    evidence = structured_result.get("evidence", [])
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                source_id = item.get("source_id")
                if isinstance(source_id, int):
                    cited_source_ids.add(source_id)

    invalid_source_ids = sorted(cited_source_ids - valid_source_ids)
    missing_citation = bool(sources) and not cited_source_ids
    status = "pass" if not missing_citation and not invalid_source_ids else "fail"

    return {
        "status": status,
        "cited_source_ids": sorted(cited_source_ids),
        "invalid_source_ids": invalid_source_ids,
        "missing_citation": missing_citation,
        "source_count": len(sources),
    }


def generate_followup_recommendation(
    case_description: str,
    guideline_result: dict[str, Any],
    rag_chain,
    nodule_type: str | None = None,
) -> dict[str, Any]:
    """结构化生成肺结节随访建议，并自动补充引用自检。"""
    resolved_nodule_type = nodule_type or classify_nodule_type(case_description)
    messages = [
        {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": FOLLOWUP_PROMPT_TEMPLATE.format(
                case_description=case_description,
                nodule_type=resolved_nodule_type,
                context=guideline_result.get("context", "（未检索到相关参考资料）"),
            ),
        },
    ]

    client = rag_chain._get_client()
    response = client.chat.completions.create(
        model=getattr(rag_chain, "model", DEEPSEEK_MODEL),
        messages=messages,
        temperature=0.2,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    structured = _parse_structured_json(content)
    structured.setdefault("nodule_type", resolved_nodule_type)
    structured.setdefault("recommendation", "证据不足，无法生成明确随访建议。")
    structured.setdefault("followup_interval", "证据不足")
    structured.setdefault("risk_level", "证据不足")
    structured.setdefault("rationale", "参考资料不足。")
    structured.setdefault("evidence", [])
    structured.setdefault("disclaimer", "以下内容仅供参考，不能替代专业医师的诊断和建议。")
    structured["citation_check"] = self_check_citations(
        structured,
        guideline_result.get("sources", []),
    )
    return structured


def _build_followup_query(case_description: str, nodule_type: str) -> str:
    type_labels = {
        "solid": "实性肺结节",
        "pure-ground-glass": "纯磨玻璃肺结节",
        "part-solid": "部分实性肺结节",
        "multiple": "多发肺结节",
        "unknown": "肺结节",
    }
    return f"{type_labels.get(nodule_type, '肺结节')} 随访 管理 指南 {case_description}"


def retrieve_guideline_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph兼容的检索节点。"""
    nodule_type = state.get("nodule_type") or classify_nodule_type(state["case_description"])
    query = state.get("guideline_query") or _build_followup_query(
        state["case_description"],
        nodule_type,
    )
    return {
        "nodule_type": nodule_type,
        "guideline_result": retrieve_guideline(
            query=query,
            vector_store=state["vector_store"],
            top_k=state.get("top_k", RETRIEVAL_TOP_K),
        ),
    }


def classify_nodule_type_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph兼容的结节类型判断节点。"""
    return {"nodule_type": classify_nodule_type(state["case_description"])}


def generate_followup_recommendation_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph兼容的结构化随访建议节点。"""
    recommendation = generate_followup_recommendation(
        case_description=state["case_description"],
        guideline_result=state["guideline_result"],
        rag_chain=state["rag_chain"],
        nodule_type=state.get("nodule_type"),
    )
    return {"recommendation": recommendation}


def self_check_citations_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph兼容的引用自检节点。"""
    recommendation = dict(state["recommendation"])
    recommendation["citation_check"] = self_check_citations(
        recommendation,
        state["guideline_result"].get("sources", []),
    )
    return {"recommendation": recommendation}


def run_followup_tool_flow(
    case_description: str,
    vector_store,
    rag_chain,
    top_k: int = RETRIEVAL_TOP_K,
) -> dict[str, Any]:
    """完成检索、结节类型判断、随访建议生成和引用自检流程。"""
    nodule_type = classify_nodule_type(case_description)
    query = _build_followup_query(case_description, nodule_type)
    guideline_result = retrieve_guideline(query, vector_store, top_k=top_k)
    recommendation = generate_followup_recommendation(
        case_description=case_description,
        guideline_result=guideline_result,
        rag_chain=rag_chain,
        nodule_type=nodule_type,
    )
    recommendation["citation_check"] = self_check_citations(
        recommendation,
        guideline_result["sources"],
    )

    return {
        "case_description": case_description,
        "nodule_type": nodule_type,
        "guideline_result": guideline_result,
        "recommendation": recommendation,
        "tool_schemas": get_followup_tool_schemas(),
        "tool_trace": [
            {
                "step": "retrieve_guideline",
                "query": query,
                "retrieved_count": guideline_result["retrieved_count"],
            },
            {"step": "classify_nodule_type", "nodule_type": nodule_type},
            {"step": "generate_followup_recommendation"},
            {
                "step": "self_check_citations",
                "status": recommendation["citation_check"]["status"],
            },
        ],
    }
