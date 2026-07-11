"""Retrieval interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.schema import EvidenceItem, RetrievalHit


class Retriever(ABC):
    """Base retriever interface."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Return ranked evidence hits."""


def cite(evidence: EvidenceItem) -> str:
    page = ""
    if evidence.page_start is not None:
        page = f", p.{evidence.page_start}"
        if evidence.page_end and evidence.page_end != evidence.page_start:
            page = f", p.{evidence.page_start}-{evidence.page_end}"
    return f"{evidence.source_file}{page}"
