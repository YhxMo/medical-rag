"""Qdrant-backed dense retrieval with the project's BM25 hybrid sidecar."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.embedding.base import EmbeddingProvider
from src.indexer.common import (
    bm25_candidates,
    build_bm25_index,
    fuse_hybrid_candidates,
    read_evidence,
    write_evidence,
)
from src.schema import EvidenceItem, RetrievalHit


class QdrantIndexStore:
    """Persist dense vectors and evidence payloads in Qdrant.

    Dense and BM25 candidates are fused with reciprocal rank fusion.
    """

    def __init__(
        self,
        index_dir: str | Path,
        *,
        collection_name: str = "medical_rag_evidence",
        local_path: str | Path | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.collection_name = collection_name
        self.local_path = Path(local_path) if local_path else self.index_dir.parent / "qdrant"
        self.bm25_path = self.index_dir / "bm25.pkl"
        self.evidence_path = self.index_dir / "evidence.jsonl"
        self._client = None

    def build(self, evidence: list[EvidenceItem], embedding_provider: EmbeddingProvider) -> None:
        """Replace the collection contents and rebuild the BM25 sidecar."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        client, models = self._connection()

        if not evidence:
            if client.collection_exists(self.collection_name):
                self._clear_collection_points(client, models)
            self.bm25_path.unlink(missing_ok=True)
            write_evidence(self.evidence_path, evidence)
            return

        vectors = np.asarray(
            embedding_provider.embed_texts([item.content for item in evidence]),
            dtype="float32",
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(evidence):
            raise ValueError("Embeddings must be a 2D matrix with one row per evidence item.")

        self._replace_collection(client, models, vector_size=vectors.shape[1])
        points = [
            models.PointStruct(
                id=self._point_id(ordinal, item),
                vector=vector.tolist(),
                payload={**asdict(item), "_ordinal": ordinal},
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
    ) -> list[RetrievalHit]:
        evidence = self.load_evidence()
        if not evidence or final_top_k <= 0:
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
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _connection(self):
        from qdrant_client import QdrantClient, models

        if self._client is None:
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self.local_path))
        return self._client, models

    def _replace_collection(self, client: Any, models: Any, *, vector_size: int) -> None:
        if not client.collection_exists(self.collection_name):
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
            return

        current_size = client.get_collection(self.collection_name).config.params.vectors.size
        if current_size != vector_size:
            client.delete_collection(self.collection_name)
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
        else:
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
        response = client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return list(response.points)

    def _point_id(self, ordinal: int, item: EvidenceItem) -> str:
        key = f"{self.collection_name}:{ordinal}:{item.evidence_id}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))
