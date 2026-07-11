"""In-memory hybrid retrieval for MVP tests and smoke runs."""
from __future__ import annotations

import math

from src.embedding.base import EmbeddingProvider
from src.schema import EvidenceItem, RetrievalHit


class InMemoryHybridRetriever:
    """Combine dense cosine scores and simple lexical overlap with RRF-like scoring."""

    def __init__(self, evidence: list[EvidenceItem], embedding_provider: EmbeddingProvider) -> None:
        self.evidence = evidence
        self.embedding_provider = embedding_provider
        self.embeddings = embedding_provider.embed_texts([item.content for item in evidence])

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        query_embedding = self.embedding_provider.embed_query(query)
        query_terms = set(_terms(query))
        scored: list[tuple[EvidenceItem, float, dict[str, float]]] = []
        for item, embedding in zip(self.evidence, self.embeddings):
            dense = _cosine(query_embedding, embedding)
            lexical = _lexical_overlap(query_terms, item.content)
            score = dense + lexical
            scored.append((item, score, {"dense": dense, "lexical": lexical}))

        scored.sort(key=lambda row: row[1], reverse=True)
        return [
            RetrievalHit(evidence=item, score=score, rank=index + 1, scores=scores)
            for index, (item, score, scores) in enumerate(scored[:top_k])
        ]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return dot / (left_norm * right_norm)


def _lexical_overlap(query_terms: set[str], content: str) -> float:
    if not query_terms:
        return 0.0
    content_terms = set(_terms(content))
    return len(query_terms & content_terms) / len(query_terms)


def _terms(text: str) -> list[str]:
    ascii_terms = "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
    if ascii_terms:
        return ascii_terms
    return [text[index : index + 2] for index in range(max(0, len(text) - 1))]
