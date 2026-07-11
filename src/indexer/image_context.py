"""Build retrieval context for image caption evidence."""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from src.schema import ExerciseQAChunk, ImageAsset, ImageCaption, PageDocument, TextChunk


_FIGURE_CAPTION_RE = re.compile(r"(图\s*\d+(?:[-－]\d+)?[^\n]{0,80}|表\s*\d+(?:[-－]\d+)?[^\n]{0,80})")


def build_image_caption_context(
    asset: ImageAsset,
    *,
    pages: Iterable[PageDocument],
    text_chunks: Iterable[TextChunk] = (),
    exercise_chunks: Iterable[ExerciseQAChunk] = (),
    max_chars: int = 900,
    max_chunks: int = 2,
) -> dict:
    """Collect page, heading, figure-caption, and nearby text context."""

    page_by_number = {page.page_number: page for page in pages}
    page = page_by_number.get(asset.page_number)
    overlapping_text = [
        chunk
        for chunk in text_chunks
        if chunk.page_start <= asset.page_number <= chunk.page_end
    ][:max_chunks]
    overlapping_exercises = [
        chunk
        for chunk in exercise_chunks
        if _page_overlaps_exercise(asset.page_number, chunk)
    ][:max_chunks]

    heading_path = _best_heading_path(overlapping_text)
    figure_captions = _extract_figure_captions(page.text if page else "")
    nearby_text = _join_snippets(
        [chunk.content for chunk in overlapping_text]
        + [_exercise_snippet(chunk) for chunk in overlapping_exercises]
        + ([page.text] if page and not overlapping_text and not overlapping_exercises else []),
        max_chars=max_chars,
    )

    return {
        "image_kind": asset.kind,
        "page_number": asset.page_number,
        "heading_path": list(heading_path),
        "figure_captions": figure_captions,
        "nearby_text": nearby_text,
        "source_file": asset.source_file,
    }


def build_contextual_caption_prompt(base_prompt: str | None, context: dict) -> str:
    """Create a VLM prompt that includes source context without asking it to hallucinate."""

    lines = [
        base_prompt or "请用中文描述这张医学教材图片。",
        "",
        "请结合下面的页码、章节、图注候选和附近正文，生成中文医学描述。",
        "重点说明：影像/图片类型、解剖部位、病变或征象、图中标注、与正文知识点的关系。",
        "不要编造上下文没有支持的诊断结论；不确定时说明可能性。",
        f"来源：{context.get('source_file', '')} 第{context.get('page_number', '')}页",
        f"图像类型：{context.get('image_kind', '')}",
    ]
    heading_path = context.get("heading_path") or []
    if heading_path:
        lines.append("章节：" + " > ".join(str(item) for item in heading_path))
    figure_captions = context.get("figure_captions") or []
    if figure_captions:
        lines.append("图注候选：" + "；".join(str(item) for item in figure_captions))
    nearby_text = context.get("nearby_text") or ""
    if nearby_text:
        lines.append("附近正文：" + str(nearby_text))
    return "\n".join(lines)


def attach_context_to_caption(caption: ImageCaption, context: dict) -> ImageCaption:
    """Return a caption with image retrieval context stored in metadata."""

    return replace(caption, metadata={**caption.metadata, **context})


def _page_overlaps_exercise(page_number: int, chunk: ExerciseQAChunk) -> bool:
    starts = [value for value in (chunk.page_question, chunk.page_answer) if value is not None]
    if not starts:
        return False
    start = min(starts)
    end = max(starts)
    return start <= page_number <= end


def _best_heading_path(chunks: list[TextChunk]) -> tuple[str, ...]:
    for chunk in chunks:
        if chunk.heading_path:
            return chunk.heading_path
    return ()


def _extract_figure_captions(text: str, *, limit: int = 5) -> list[str]:
    captions = []
    for match in _FIGURE_CAPTION_RE.finditer(text):
        caption = _compact(match.group(1))
        if caption and caption not in captions:
            captions.append(caption)
        if len(captions) >= limit:
            break
    return captions


def _join_snippets(snippets: list[str], *, max_chars: int) -> str:
    output: list[str] = []
    remaining = max_chars
    for snippet in snippets:
        compacted = _compact(snippet)
        if not compacted:
            continue
        if len(compacted) > remaining:
            compacted = compacted[:remaining].rstrip()
        output.append(compacted)
        remaining -= len(compacted)
        if remaining <= 0:
            break
    return "\n".join(output)


def _exercise_snippet(chunk: ExerciseQAChunk) -> str:
    parts = [f"题目：{chunk.question_text}"]
    if chunk.options:
        parts.append("选项：" + " ".join(chunk.options))
    if chunk.answer:
        parts.append(f"答案：{chunk.answer}")
    if chunk.explanation:
        parts.append(f"解析：{chunk.explanation}")
    return "\n".join(parts)


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
