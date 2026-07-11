"""Convert heterogeneous parsed content into unified evidence records."""
from __future__ import annotations

from src.schema import EvidenceItem, ExerciseQAChunk, ImageCaption, TextChunk


def evidence_from_text_chunk(chunk: TextChunk) -> EvidenceItem:
    metadata = {
        "chunk_id": chunk.chunk_id,
        "heading_path": list(chunk.heading_path),
        "chunk_strategy": chunk.chunk_strategy,
        "nearby_image_ids": list(chunk.nearby_image_ids),
        "nearby_caption_ids": list(chunk.nearby_caption_ids),
        "needs_ocr": chunk.needs_ocr,
        **chunk.metadata,
    }
    return EvidenceItem(
        evidence_id=chunk.chunk_id,
        evidence_type="text",
        source_file=chunk.source_file,
        content=chunk.content,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        metadata=metadata,
    )


def evidence_from_image_caption(caption: ImageCaption) -> EvidenceItem:
    metadata = {
        "caption_id": caption.caption_id,
        "image_id": caption.image_id,
        "is_ocr_text": caption.is_ocr_text,
        **caption.metadata,
    }
    content_parts = [
        f"图像描述：{caption.caption}",
        f"来源：{caption.source_file} 第{caption.page_number}页",
    ]
    image_kind = metadata.get("image_kind")
    if image_kind:
        content_parts.append(f"图像类型：{image_kind}")
    heading_path = metadata.get("heading_path") or []
    if heading_path:
        content_parts.append("章节：" + " > ".join(str(item) for item in heading_path))
    figure_captions = metadata.get("figure_captions") or []
    if figure_captions:
        content_parts.append("图注：" + "；".join(str(item) for item in figure_captions))
    nearby_text = metadata.get("nearby_text")
    if nearby_text:
        content_parts.append(f"附近正文：{nearby_text}")
    return EvidenceItem(
        evidence_id=caption.caption_id,
        evidence_type="image_caption",
        source_file=caption.source_file,
        content="\n".join(content_parts),
        page_start=caption.page_number,
        page_end=caption.page_number,
        metadata=metadata,
    )


def evidence_from_exercise_qa(chunk: ExerciseQAChunk) -> EvidenceItem:
    parts = [f"题目：{chunk.question_text}"]
    if chunk.options:
        parts.append("选项：\n" + "\n".join(chunk.options))
    if chunk.answer:
        parts.append(f"答案：{chunk.answer}")
    if chunk.explanation:
        parts.append(f"解析：{chunk.explanation}")

    metadata = {
        "question_id": chunk.question_id,
        "chapter": chunk.chapter,
        "question_number": chunk.question_number,
        "question_type": chunk.question_type,
        "question_text": chunk.question_text,
        "options": list(chunk.options),
        "answer": chunk.answer,
        "explanation": chunk.explanation,
        "page_question": chunk.page_question,
        "page_answer": chunk.page_answer,
        "related_image_ids": list(chunk.related_image_ids),
        "chunk_strategy": chunk.chunk_strategy,
        **chunk.metadata,
    }
    return EvidenceItem(
        evidence_id=chunk.question_id,
        evidence_type="exercise_qa",
        source_file=chunk.source_file,
        content="\n".join(parts),
        page_start=chunk.page_question,
        page_end=chunk.page_answer or chunk.page_question,
        metadata=metadata,
    )


def build_evidence_items(
    text_chunks: list[TextChunk] | None = None,
    captions: list[ImageCaption] | None = None,
    exercise_chunks: list[ExerciseQAChunk] | None = None,
) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    evidence.extend(evidence_from_text_chunk(chunk) for chunk in text_chunks or [])
    evidence.extend(evidence_from_image_caption(caption) for caption in captions or [])
    evidence.extend(evidence_from_exercise_qa(chunk) for chunk in exercise_chunks or [])
    return evidence
