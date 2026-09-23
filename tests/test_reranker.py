from src.reranker.base import NoopReranker
from src.schema import EvidenceItem, RetrievalHit


def test_noop_reranker_preserves_order_and_limits_top_k():
    hits = [
        RetrievalHit(EvidenceItem("e1", "text", "a.pdf", "a"), 0.2, 1),
        RetrievalHit(EvidenceItem("e2", "text", "a.pdf", "b"), 0.1, 2),
    ]

    reranked = NoopReranker().rerank("query", hits, top_k=1)

    assert len(reranked) == 1
    assert reranked[0].evidence.evidence_id == "e1"
