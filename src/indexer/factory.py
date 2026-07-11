"""Configuration-driven construction of persistent evidence stores."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.indexer.base import EvidenceStore
from src.indexer.qdrant_store import QdrantIndexStore
from src.indexer.store import IndexStore


def create_index_store(settings: Any, *, index_dir: str | Path | None = None) -> EvidenceStore:
    """Build the configured vector-store backend.

    Qdrant is the default; FAISS remains available as a lightweight baseline
    through ``active.vector_store: faiss``.
    """
    backend = str(
        settings.get(
            "active",
            "vector_store",
            default=settings.get("vector_store", "backend", default="qdrant"),
        )
    ).strip().lower()
    resolved_index_dir = Path(index_dir or settings.get("paths", "index_dir", default="artifacts/index"))

    if backend == "faiss":
        return IndexStore(resolved_index_dir)
    if backend != "qdrant":
        raise ValueError("Unsupported active.vector_store '%s'. Use 'qdrant' or 'faiss'." % backend)

    artifact_dir = Path(settings.get("paths", "artifact_dir", default="artifacts"))
    return QdrantIndexStore(
        resolved_index_dir,
        collection_name=str(settings.get("vector_store", "qdrant", "collection", default="medical_rag_evidence")),
        url=settings.get("vector_store", "qdrant", "url", default="") or None,
        api_key=settings.get("vector_store", "qdrant", "api_key", default="") or None,
        local_path=settings.get("vector_store", "qdrant", "local_path", default=str(artifact_dir / "qdrant")),
    )
