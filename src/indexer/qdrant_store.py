"""Qdrant-backed dense retrieval with the project's BM25 hybrid sidecar."""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from src.embedding.base import EmbeddingProvider
from src.indexer.common import (
    bm25_candidates,
    build_bm25_index,
    evidence_to_json,
    fuse_hybrid_candidates,
    read_evidence,
    retrieval_text,
    write_evidence,
)
from src.schema import EvidenceItem, RetrievalHit


class QdrantIndexStore:
    """Persist dense vectors and evidence payloads in Qdrant.

    BM25 remains a local sidecar so the existing RRF fusion, ablations, and
    evidence-type weighting retain their current behavior.
    """

    def __init__(
        self,
        index_dir: str | Path,
        *,
        collection_name: str = "medical_rag_evidence",
        url: str | None = None,
        api_key: str | None = None,
        local_path: str | Path | None = None,
        client: Any | None = None,
        models_module: Any | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.collection_name = collection_name
        self.url = url.strip() if url else None
        self.api_key = api_key or None
        self.local_path = Path(local_path) if local_path else self.index_dir.parent / "qdrant"
        self.bm25_path = self.index_dir / "bm25.pkl"
        self.evidence_path = self.index_dir / "evidence.jsonl"
        self._client = client
        self._models_module = models_module

    def build(self, evidence: list[EvidenceItem], embedding_provider: EmbeddingProvider) -> None:
        """Replace the collection contents and rebuild the BM25 sidecar."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        client, models = self._connection()

        if not evidence:
            if client.collection_exists(self.collection_name):
                self._clear_collection_points(client, models)
            write_evidence(self.evidence_path, evidence)
            return

        vectors = np.asarray(embedding_provider.embed_texts([retrieval_text(item) for item in evidence]), dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(evidence):
            raise ValueError("Embeddings must be a 2D matrix with one row per evidence item.")

        self._replace_collection(client, models, vector_size=vectors.shape[1])
        points = [
            models.PointStruct(
                id=self._point_id(ordinal, item),
                vector=vector.tolist(),
                payload={**evidence_to_json(item), "_ordinal": ordinal},
            )
            for ordinal, (item, vector) in enumerate(zip(evidence, vectors, strict=True))
        ]
        client.upsert(collection_name=self.collection_name, points=points, wait=True)
        build_bm25_index(self.bm25_path, evidence)
        write_evidence(self.evidence_path, evidence)

    def load_evidence(self) -> list[EvidenceItem]:
        return read_evidence(self.evidence_path)

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
        evidence = self.load_evidence()
        if not evidence:
            return []

        dense_limit = min(max(0, dense_top_k), len(evidence))
        dense: list[tuple[int, float]] = []
        if dense_limit:
            query_vector = list(embedding_provider.embed_query(query))
            for point in self._query_points(query_vector, dense_limit):
                payload = getattr(point, "payload", None) or {}
                ordinal = payload.get("_ordinal")
                if isinstance(ordinal, int):
                    dense.append((ordinal, float(point.score)))

        bm25_limit = min(max(0, bm25_top_k), len(evidence))
        bm25 = bm25_candidates(self.bm25_path, query, limit=bm25_limit) if bm25_limit else []
        return fuse_hybrid_candidates(
            evidence,
            dense=dense,
            bm25=bm25,
            final_top_k=final_top_k,
            evidence_type_weights=evidence_type_weights,
        )

    def create_empty(self, index_dir: str | Path) -> "QdrantIndexStore":
        """Create an isolated collection for a temporary ablation index."""
        new_index_dir = Path(index_dir)
        suffix = hashlib.sha1(str(new_index_dir.resolve()).encode("utf-8")).hexdigest()[:12]
        return QdrantIndexStore(
            new_index_dir,
            collection_name=f"{self.collection_name}_{suffix}",
            url=self.url,
            api_key=self.api_key,
            local_path=None if self.url else new_index_dir / "qdrant",
        )

    def _connection(self) -> tuple[Any, Any]:
        if self._client is not None:
            if self._models_module is None:
                raise RuntimeError("models_module is required when injecting a Qdrant client.")
            return self._client, self._models_module

        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is required for the qdrant vector-store backend. "
                "Install the project's requirements in the active environment."
            ) from exc

        if self.url:
            self._client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self.local_path))
        self._models_module = models
        return self._client, models

    def _replace_collection(self, client: Any, models: Any, *, vector_size: int) -> None:
        if not client.collection_exists(self.collection_name):
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            return

        self._clear_collection_points(client, models)

    def _clear_collection_points(self, client: Any, models: Any) -> None:
        """Remove every point while retaining the collection schema."""
        # Qdrant Local can retain old points when a collection is deleted and
        # recreated through the same client. Delete point IDs explicitly so a
        # repeated build is deterministic for both embedded and remote modes.
        while True:
            points, _ = client.scroll(
                collection_name=self.collection_name,
                limit=1024,
                with_payload=False,
                with_vectors=False,
            )
            if not points:
                break
            client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[point.id for point in points]),
                wait=True,
            )

        if client.count(collection_name=self.collection_name, exact=True).count:
            raise RuntimeError(f"Qdrant collection was not cleared: {self.collection_name}")

    def _query_points(self, query_vector: list[float], limit: int) -> list[Any]:
        client, _ = self._connection()
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return list(getattr(response, "points", response))
        return list(
            client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        )

    def _point_id(self, ordinal: int, item: EvidenceItem) -> str:
        key = f"{self.collection_name}:{ordinal}:{item.evidence_id}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))
