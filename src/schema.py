"""Shared data models for the medical RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


EvidenceType = Literal["text", "image_caption", "exercise_qa"]


@dataclass(frozen=True)
class PageDocument:
    """Text extracted from a single PDF page."""

    source_file: str
    page_number: int
    text: str
    text_length: int
    needs_ocr: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls,
        source_file: str,
        page_number: int,
        text: str,
        *,
        ocr_min_chars: int = 50,
        metadata: dict[str, Any] | None = None,
    ) -> "PageDocument":
        normalized = text.strip()
        return cls(
            source_file=source_file,
            page_number=page_number,
            text=normalized,
            text_length=len(normalized),
            needs_ocr=len(normalized) < ocr_min_chars,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class ImageAsset:
    """An extracted page image or embedded image."""

    image_id: str
    source_file: str
    page_number: int
    path: Path
    kind: Literal["page", "embedded"] = "page"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextChunk:
    """A chunk of textbook text produced by a named strategy."""

    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    heading_path: tuple[str, ...]
    content: str
    chunk_strategy: str
    nearby_image_ids: tuple[str, ...] = ()
    nearby_caption_ids: tuple[str, ...] = ()
    needs_ocr: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageCaption:
    """Caption or OCR text generated from a page/image."""

    caption_id: str
    image_id: str
    source_file: str
    page_number: int
    caption: str
    is_ocr_text: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExerciseQAChunk:
    """One structured exercise question paired with its answer/explanation."""

    question_id: str
    source_file: str
    chapter: str
    question_number: str
    question_type: str
    question_text: str
    options: tuple[str, ...] = ()
    answer: str = ""
    explanation: str = ""
    page_question: int | None = None
    page_answer: int | None = None
    related_image_ids: tuple[str, ...] = ()
    chunk_strategy: str = "exercise_qa_pair"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExerciseParseReport:
    """Structured exercise parsing result."""

    chunks: tuple[ExerciseQAChunk, ...]
    unmatched_questions: tuple[ExerciseQAChunk, ...] = ()
    unmatched_answers: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceItem:
    """Unified evidence record that can be embedded and retrieved."""

    evidence_id: str
    evidence_type: EvidenceType
    source_file: str
    content: str
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    """A retrieved evidence item with scores."""

    evidence: EvidenceItem
    score: float
    rank: int
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedAnswer:
    """Generated answer and the evidence used to produce it."""

    answer: str
    sources: tuple[EvidenceItem, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
