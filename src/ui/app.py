"""A small Gradio adapter; all query behavior lives in QueryService."""

from __future__ import annotations

import html


def format_sources(hits):
    return "\n\n".join(
        f"<details><summary>[{i}] {html.escape(h.evidence.source_file)} · "
        f"PDF 第 {h.evidence.page_start} 页 · "
        f"{'AI 图像描述' if h.evidence.evidence_type == 'image_caption' else '教材原文'}"
        f"</summary><pre>{html.escape(h.evidence.content)}</pre></details>"
        for i, h in enumerate(hits, 1)
    )


def create_app(service):
    import gradio as gr

    def run(question, image, offline):
        try:
            result = service.ask(question, image, offline=offline)
            note = "\n\n".join(result.warnings)
            return result.answer + ("\n\n" + note if note else ""), format_sources(result.sources)
        except Exception as exc:
            message = (
                str(exc) if isinstance(exc, ValueError) else "查询失败，请检查模型配置与本地索引。"
            )
            raise gr.Error(message) from None

    with gr.Blocks(title="医学影像教材学习助手", delete_cache=(60, 60)) as app:
        gr.Markdown(
            "# 医学影像教材学习助手\n混合检索 · 截图提问 · 引用溯源\n\n用于教材学习，不作患者诊断。"
        )
        question = gr.Textbox(label="问题", placeholder="输入中文或英文问题", lines=3)
        screenshot = gr.File(
            label="教材截图（启用模型时会发送至视觉服务）",
            file_types=[".png", ".jpg", ".jpeg", ".webp"],
            type="filepath",
        )
        offline = gr.Checkbox(label="仅检索教材原文（不调用 API）", value=True)
        gr.Examples(
            examples=[
                "What determines axial resolution in ultrasound?",
                "超声的轴向分辨率由什么决定？",
            ],
            inputs=question,
        )
        submit = gr.Button("检索并回答", variant="primary")
        answer = gr.Markdown()
        with gr.Accordion("教材出处与原文证据", open=True):
            sources = gr.HTML()
        submit.click(run, [question, screenshot, offline], [answer, sources], concurrency_limit=1)
    return app
