"""Data passed between document ingestion, retrieval and answering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class PageDocument:
    source_file: str
    page_number: int
    text: str

    @classmethod
    def from_text(cls, source_file, page_number, text):
        return cls(source_file, page_number, text.strip())


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    evidence_type: Literal["text", "image_caption"]
    source_file: str
    content: str
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    evidence: EvidenceItem
    score: float
    rank: int
    scores: dict[str, float] = field(default_factory=dict)
