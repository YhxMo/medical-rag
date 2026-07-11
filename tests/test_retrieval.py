from src.embedding.simple import HashEmbeddingProvider
from src.retrieval.hybrid import InMemoryHybridRetriever
from src.schema import EvidenceItem


def test_in_memory_hybrid_retriever_returns_relevant_evidence():
    evidence = [
        EvidenceItem("e1", "text", "book.pdf", "脑出血 CT 表现为高密度影", 1, 1),
        EvidenceItem("e2", "text", "book.pdf", "肺炎可见斑片状阴影", 2, 2),
    ]
    retriever = InMemoryHybridRetriever(evidence, HashEmbeddingProvider())

    hits = retriever.retrieve("脑出血 CT", top_k=1)

    assert hits[0].evidence.evidence_id == "e1"
    assert hits[0].rank == 1
