"""Sentence-transformers BGE embedding provider."""
from __future__ import annotations

from src.embedding.base import EmbeddingProvider


class BGEEmbeddingProvider(EmbeddingProvider):
    """Lazy wrapper around sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = None if device in {None, "auto"} else device
        self._model = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for BGE embeddings.") from exc
        self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model
