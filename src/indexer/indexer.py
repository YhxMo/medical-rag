"""Index builder facade."""
from __future__ import annotations

from pathlib import Path

from src.embedding.base import EmbeddingProvider
from src.indexer.base import EvidenceStore
from src.indexer.store import IndexStore
from src.schema import EvidenceItem


class Indexer:
    """Build persistent indexes from unified evidence."""

    def __init__(
        self,
        index_dir: str | Path,
        embedding_provider: EmbeddingProvider,
        *,
        store: EvidenceStore | None = None,
    ) -> None:
        self.store = store or IndexStore(index_dir)
        self.embedding_provider = embedding_provider

    def build(self, evidence: list[EvidenceItem]) -> EvidenceStore:
        self.store.build(evidence, self.embedding_provider)
        return self.store
