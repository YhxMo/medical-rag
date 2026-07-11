"""Contracts shared by persistent evidence stores."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.embedding.base import EmbeddingProvider
from src.schema import EvidenceItem, RetrievalHit


class EvidenceStore(Protocol):
    """Persistent storage contract used by indexing, retrieval, and evaluation."""

    index_dir: Path

    def build(self, evidence: list[EvidenceItem], embedding_provider: EmbeddingProvider) -> None:
        """Persist evidence and any indexes needed to search it."""

    def load_evidence(self) -> list[EvidenceItem]:
        """Return the evidence records stored by this backend."""

    def search(
        self,
        query: str,
        embedding_provider: EmbeddingProvider,
        *,
        dense_top_k: int = 10,
        bm25_top_k: int = 10,
        final_top_k: int = 5,
        evidence_type_weights: dict[str, float] | None = None,
    ) -> list[RetrievalHit]:
        """Run hybrid retrieval and return ranked evidence."""

    def create_empty(self, index_dir: str | Path) -> "EvidenceStore":
        """Create an equivalent backend suitable for a temporary rebuilt index."""
