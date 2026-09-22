"""Local-only resume demo backed by the same service as CLI queries."""

from __future__ import annotations
import json
from pathlib import Path
import tempfile
import shutil
import queue
import threading
import copy
import html
from src.application.images import registered_image


def create_resume_app(service):
    import gradio as gr

    def run(question, uploaded, offline):
        # Copy only for this request; never delete the user's original local file.
        with tempfile.TemporaryDirectory(prefix="medical-rag-upload-") as directory:
            image = None
            if uploaded:
                source = Path(uploaded)
                image = str(Path(directory) / ("upload" + source.suffix.lower()))
                shutil.copyfile(source, image)
            yield "正在检索与核对证据…", "", [], "引用待校验"
            events = queue.Queue()

            def worker():
                try:
                    result = service.ask(
                        question,
                        image,
                        offline=offline,
                        event=lambda r: events.put(("progress", copy.deepcopy(r))),
                    )
                    events.put(("final", result))
                except Exception:
                    events.put(("error", None))

            threading.Thread(target=worker, daemon=True).start()
            while True:
                state, result = events.get()
                if state == "error":
                    yield "服务未完成，请检查本地配置。", "", [], "service_error"
                    return
                if state == "final":
                    break
                source_preview = "\n".join(
                    f"[{i}] {h.evidence.source_file} · PDF 第 {h.evidence.page_start} 页"
                    for i, h in enumerate(result.sources, 1)
                )
                yield (
                    result.answer or "已检索到证据，正在生成…",
                    source_preview,
                    [],
                    "引用待校验",
                )
            sources = "\n\n".join(
                f"<details><summary>[{i}] {html.escape(h.evidence.source_file)} · PDF 第 {h.evidence.page_start} 页 "
                f"({h.evidence.evidence_type})</summary><pre>{html.escape(h.evidence.content)}</pre></details>"
                for i, h in enumerate(result.sources, 1)
            )
            gallery = []
            for path in result.image_paths:
                registered_image(path, service.registry)
                gallery.append((path, "已登记教材原图"))
            detail = {
                "status": result.status,
                "warnings": result.warnings,
                "timings": result.timings,
                "model_calls": result.calls,
                "retrieval_query": result.query,
                "citation_check": result.citations,
            }
            yield (
                result.answer,
                sources,
                gallery,
                json.dumps(detail, ensure_ascii=False, indent=2),
            )

    with gr.Blocks(title="医学影像教材学习助手", delete_cache=(60, 60)) as app:
        gr.Markdown(
            "# 医学影像教材学习助手\n图文检索 · 教材截图提问 · 来源溯源\n\n用于教材学习，不作患者影像诊断。"
        )
        with gr.Row():
            question = gr.Textbox(
                label="问题",
                placeholder="输入中文或英文问题，也可以只上传教材截图",
                lines=3,
            )
            screenshot = gr.File(
                label="教材截图（将发送至配置的视觉模型服务）",
                file_types=[".png", ".jpg", ".jpeg", ".webp"],
                type="filepath",
            )
        offline = gr.Checkbox(label="仅本地检索证据（不调用 API）", value=True)
        gr.Examples(
            examples=[
                ["What determines axial resolution in ultrasound?"],
                ["超声的轴向分辨率由什么决定？"],
                ["What is the exact measured value in my experiment?"],
            ],
            inputs=[question],
        )
        submit = gr.Button("检索并回答", variant="primary")
        answer = gr.Markdown()
        with gr.Accordion("教材出处与原文证据", open=True):
            sources = gr.Markdown()
            gallery = gr.Gallery(label="教材原图", columns=2)
        with gr.Accordion("运行详情", open=False):
            detail = gr.Code(language="json")
        submit.click(
            run,
            [question, screenshot, offline],
            [answer, sources, gallery, detail],
            concurrency_limit=1,
        )
    return app
