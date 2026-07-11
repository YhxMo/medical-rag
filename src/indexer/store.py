"""Legacy FAISS + BM25 index store kept for lightweight local comparisons."""
from __future__ import annotations

from pathlib import Path

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


class IndexStore:
    """Persist and query dense FAISS and sparse BM25 indexes."""

    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self.faiss_path = self.index_dir / "dense.faiss"
        self.bm25_path = self.index_dir / "bm25.pkl"
        self.evidence_path = self.index_dir / "evidence.jsonl"

    def build(self, evidence: list[EvidenceItem], embedding_provider: EmbeddingProvider) -> None:
        import faiss

        self.index_dir.mkdir(parents=True, exist_ok=True)
        if not evidence:
            write_evidence(self.evidence_path, evidence)
            return

        embeddings = np.array(embedding_provider.embed_texts([item.content for item in evidence]), dtype="float32")
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix.")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(self.faiss_path))

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
            import faiss

            index = faiss.read_index(str(self.faiss_path))
            query_vector = np.array([embedding_provider.embed_query(query)], dtype="float32")
            faiss.normalize_L2(query_vector)
            dense_scores, dense_indices = index.search(query_vector, dense_limit)
            for idx, score in zip(dense_indices[0], dense_scores[0]):
                if idx < 0:
                    continue
                dense.append((int(idx), float(score)))

        bm25_limit = min(max(0, bm25_top_k), len(evidence))
        bm25 = bm25_candidates(self.bm25_path, query, limit=bm25_limit) if bm25_limit else []
        return fuse_hybrid_candidates(
            evidence,
            dense=dense,
            bm25=bm25,
            final_top_k=final_top_k,
            evidence_type_weights=evidence_type_weights,
        )

    def create_empty(self, index_dir: str | Path) -> "IndexStore":
        return IndexStore(index_dir)
