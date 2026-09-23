"""Reranker interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.schema import RetrievalHit


class Reranker(ABC):
    """Base reranker interface."""

    @abstractmethod
    def rerank(
        self, query: str, hits: list[RetrievalHit], *, top_k: int | None = None
    ) -> list[RetrievalHit]:
        """Return reranked hits."""


class NoopReranker(Reranker):
    """Keep original retrieval order."""

    def rerank(
        self, query: str, hits: list[RetrievalHit], *, top_k: int | None = None
    ) -> list[RetrievalHit]:
        return hits[:top_k] if top_k is not None else hits
