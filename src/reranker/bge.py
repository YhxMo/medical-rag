"""BGE cross-encoder reranker."""
from __future__ import annotations

from dataclasses import replace

from src.reranker.base import Reranker
from src.schema import RetrievalHit
from src.indexer.common import retrieval_text


class BGEReranker(Reranker):
    """Lazy wrapper around sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = None if device in {None, "auto"} else device
        self._model = None

    def rerank(self, query: str, hits: list[RetrievalHit], *, top_k: int | None = None) -> list[RetrievalHit]:
        if not hits:
            return []
        model = self._load_model()
        pairs = [(query, retrieval_text(hit.evidence)) for hit in hits]
        scores = model.predict(pairs)
        reranked = [
            replace(
                hit,
                score=float(score),
                rank=0,
                scores={**hit.scores, "rerank": float(score)},
            )
            for hit, score in zip(hits, scores)
        ]
        reranked.sort(key=lambda hit: hit.score, reverse=True)
        limited = reranked[:top_k] if top_k is not None else reranked
        return [replace(hit, rank=index + 1) for index, hit in enumerate(limited)]

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for BGE reranking.") from exc
        self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model
