"""Deterministic lightweight embeddings for tests and offline smoke runs."""

from __future__ import annotations

import hashlib
import math

from src.embedding.base import EmbeddingProvider


class HashEmbeddingProvider(EmbeddingProvider):
    """Tiny deterministic embedding provider without model dependencies."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _tokens(text)
        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def _tokens(text: str) -> list[str]:
    compact = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    words = compact.split()
    if words:
        return words
    return [text[index : index + 2] for index in range(max(0, len(text) - 1))]
