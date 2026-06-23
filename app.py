"""Gradio Web 交互界面"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import gradio as gr

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from guideline_tools import run_followup_tool_flow
from ingest import load_vector_store, build_vector_store
from rag_chain import RAGChain, query_with_rag

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 初始化向量存储（若不存在则自动构建）
try:
    vector_store = load_vector_store()
    logger.info("向量知识库加载成功")
except Exception:
    logger.warning("向量知识库不存在，请先构建。将在首次问答时尝试构建。")
    vector_store = None

rag_chain = RAGChain({
    "api_key": DEEPSEEK_API_KEY,
    "base_url": DEEPSEEK_BASE_URL,
    "model": DEEPSEEK_MODEL,
})


def ensure_vector_store_ready():
    """确保向量知识库已加载或构建。"""
    global vector_store

    if vector_store is not None:
        return vector_store, None

    try:
        vector_store = load_vector_store()
    except Exception:
        try:
            vector_store = build_vector_store()
        except Exception as e:
            return None, f"❌ 向量知识库初始化失败：{str(e)}\n请确保 data/pdfs/ 目录下有PDF文件后重启。"
    return vector_store, None


def chat_with_rag(
    message: str,
    history: list[list[str]],
    top_k: int = 4,
) -> str:
    """RAG问答处理函数。"""
    global vector_store

    if not message.strip():
        return "请输入问题。"

    store, error = ensure_vector_store_ready()
    if error:
        return error

    try:
        result = query_with_rag(message, store, rag_chain, top_k=top_k)
    except Exception as e:
        logger.error("问答出错: %s", e)
        return f"❌ 生成回答时出错：{str(e)}"

    # 格式化输出
    answer_parts = [result["answer"]]

    if result["sources"]:
        answer_parts.append("\n\n---")
        answer_parts.append(f"📚 **引用来源（共 {len(result['sources'])} 篇）：**")
        for i, src in enumerate(result["sources"], 1):
            answer_parts.append(
                f"\n**[{i}]** `{src['source']}` (片段{src['chunk_index']})\n"
                f"> {src['content']}"
            )

    return "\n".join(answer_parts)


def followup_with_tools(case_description: str, top_k: int = 4) -> str:
    """通过工具化流程生成肺结节随访建议。"""
    if not case_description.strip():
        return "请输入肺结节影像描述。"

    store, error = ensure_vector_store_ready()
    if error:
        return error

    try:
        result = run_followup_tool_flow(
            case_description=case_description,
            vector_store=store,
            rag_chain=rag_chain,
            top_k=top_k,
        )
    except Exception as e:
        logger.error("随访建议生成出错: %s", e)
        return f"❌ 随访建议生成出错：{str(e)}"

    recommendation = result["recommendation"]
    citation_check = recommendation.get("citation_check", {})

    parts = [
        "### 结构化随访建议",
        f"- **结节类型**: {recommendation.get('nodule_type', result['nodule_type'])}",
        f"- **随访间隔**: {recommendation.get('followup_interval', '证据不足')}",
        f"- **风险/管理级别**: {recommendation.get('risk_level', '证据不足')}",
        "",
        "**建议**",
        recommendation.get("recommendation", "证据不足，无法生成明确随访建议。"),
        "",
        "**依据**",
        recommendation.get("rationale", "参考资料不足。"),
    ]

    evidence = recommendation.get("evidence", [])
    if evidence:
        parts.append("")
        parts.append("**证据摘录**")
        for item in evidence:
            if isinstance(item, dict):
                parts.append(
                    f"- 来源{item.get('source_id', '?')}: {item.get('summary', '')}"
                )

    parts.extend([
        "",
        "**引用自检**",
        (
            f"- 状态: {citation_check.get('status', 'unknown')}; "
            f"已引用来源: {citation_check.get('cited_source_ids', [])}; "
            f"无效来源: {citation_check.get('invalid_source_ids', [])}"
        ),
    ])

    if result["guideline_result"]["sources"]:
        parts.append("")
        parts.append("**检索来源**")
        for source in result["guideline_result"]["sources"]:
            parts.append(
                f"- 来源{source['source_id']}: `{source['source']}` "
                f"(片段{source['chunk_index']})"
            )

    parts.append("")
    parts.append(recommendation.get("disclaimer", "以下内容仅供参考，不能替代专业医师的诊断和建议。"))
    return "\n".join(parts)


def build_knowledge_base() -> str:
    """构建/重建向量知识库的处理函数。"""
    global vector_store
    try:
        vector_store = build_vector_store()
        collection_data = vector_store.get()
        count = len(collection_data.get("ids", [])) if collection_data else 0
        return f"✅ 知识库构建成功！共索引 {count} 条记录。"
    except Exception as e:
        return f"❌ 构建失败：{str(e)}"


# 构建 Gradio 界面
with gr.Blocks(title="医学影像 RAG 问答助手") as demo:
    gr.Markdown("""
    # 🏥 医学影像 RAG 问答助手

    基于检索增强生成（RAG）技术的医学影像知识问答系统。
    知识库包含医学影像教材和指南内容，回答自带来源引用。

    **示例问题：**
    - 什么是肺磨玻璃结节（GGN）？
    - 肺结节如何根据大小进行随访管理？
    - 胸部CT的肺窗和纵隔窗有什么不同？
    """)

    with gr.Tab("💬 智能问答"):
        gr.ChatInterface(
            fn=chat_with_rag,
            title="",
            description="在下方输入您的医学影像相关问题",
            examples=[
                ["什么是肺磨玻璃结节（GGN）？", 4],
                ["Fleischner指南如何对肺结节进行风险分层？", 4],
                ["部分实性结节和纯磨玻璃结节的区别是什么？", 4],
                ["哪些影像学特征提示肺结节为恶性？", 4],
            ],
            additional_inputs=[
                gr.Slider(
                    minimum=1, maximum=10, value=4, step=1,
                    label="检索文档数量 (top_k)",
                ),
            ],
        )

    with gr.Tab("📋 随访建议"):
        gr.Markdown("### 肺结节随访建议结构化生成")
        case_input = gr.Textbox(
            label="影像描述",
            lines=5,
            placeholder="例：右上肺8mm部分实性结节，实性成分约3mm，无既往肿瘤史。",
        )
        followup_top_k = gr.Slider(
            minimum=1, maximum=10, value=4, step=1,
            label="检索文档数量 (top_k)",
        )
        followup_btn = gr.Button("生成随访建议", variant="primary")
        followup_output = gr.Markdown()
        followup_btn.click(
            fn=followup_with_tools,
            inputs=[case_input, followup_top_k],
            outputs=followup_output,
        )

    with gr.Tab("⚙️ 知识库管理"):
        gr.Markdown("### 管理向量知识库")
        gr.Markdown("将医学PDF文档放入 `data/pdfs/` 目录后，点击下方按钮构建/重建知识库。")
        build_btn = gr.Button("🔨 构建/重建知识库", variant="primary")
        build_status = gr.Textbox(label="状态", lines=3)
        build_btn.click(fn=build_knowledge_base, outputs=build_status)

    with gr.Tab("ℹ️ 关于"):
        gr.Markdown("""
        ### 技术栈
        - **嵌入模型**: BAAI/bge-small-zh-v1.5（本地运行）
        - **向量数据库**: ChromaDB
        - **大语言模型**: DeepSeek API
        - **框架**: LangChain + Gradio

        ### 注意事项
        - 本系统回答仅供参考，不能替代专业医师诊断
        - 知识库内容来源于医学教材和指南PDF
        - DeepSeek API需要有效的API Key（环境变量 `DEEPSEEK_API_KEY`）
        """)


def main():
    """启动Gradio服务"""
    import os
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("[WARNING] 请设置环境变量 DEEPSEEK_API_KEY")
        print("   PowerShell: $env:DEEPSEEK_API_KEY = 'your-key'")
        print("   Bash: export DEEPSEEK_API_KEY='your-key'")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css="""
        .source-box { font-size: 0.85em; background: #f5f5f5; padding: 8px; border-radius: 6px; margin: 4px 0; }
        """,
    )


if __name__ == "__main__":
    main()
