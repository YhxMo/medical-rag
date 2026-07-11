"""End-to-end RAG pipeline orchestration."""
from __future__ import annotations

from src.generator.generator import ExtractiveGenerator
from src.reranker.base import NoopReranker, Reranker
from src.retrieval.hybrid import InMemoryHybridRetriever
from src.schema import GeneratedAnswer


class RagPipeline:
    """Run retrieval and generation."""

    def __init__(
        self,
        retriever: InMemoryHybridRetriever,
        generator: ExtractiveGenerator | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator or ExtractiveGenerator()
        self.reranker = reranker or NoopReranker()

    def answer(self, question: str, *, top_k: int = 5) -> GeneratedAnswer:
        hits = self.retriever.retrieve(question, top_k=top_k)
        hits = self.reranker.rerank(question, hits, top_k=top_k)
        return self.generator.generate(question, hits)
