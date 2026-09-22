"""Reuse version-matched text vectors when adding visual evidence."""

from __future__ import annotations
import hashlib
from src.embedding.base import EmbeddingProvider


def text_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedEmbeddingProvider(EmbeddingProvider):
    def __init__(self, provider, vectors):
        self.provider = provider
        self.vectors = vectors
        self.reused = 0
        self.computed = 0

    def embed_texts(self, texts):
        missing = {}
        for text in texts:
            key = text_key(text)
            if key not in self.vectors:
                missing[key] = text
        if missing:
            values = self.provider.embed_texts(list(missing.values()))
            if len(values) != len(missing):
                raise ValueError("Embedding count mismatch")
            self.vectors.update(zip(missing, values))
        self.computed += len(missing)
        self.reused += len(texts) - len(missing)
        return [list(self.vectors[text_key(text)]) for text in texts]

    def embed_query(self, query):
        return self.provider.embed_query(query)
