from src.embedding.simple import HashEmbeddingProvider
from src.generator.generator import ExtractiveGenerator
from src.pipeline.pipeline import RagPipeline
from src.retrieval.hybrid import InMemoryHybridRetriever
from src.schema import EvidenceItem


def test_pipeline_returns_answer_with_sources():
    evidence = [EvidenceItem("e1", "text", "book.pdf", "脑出血 CT 表现为高密度影", 1, 1)]
    retriever = InMemoryHybridRetriever(evidence, HashEmbeddingProvider())
    pipeline = RagPipeline(retriever, ExtractiveGenerator())

    answer = pipeline.answer("脑出血CT表现是什么？", top_k=1)

    assert "来源" in answer.answer
    assert answer.sources[0].evidence_id == "e1"
