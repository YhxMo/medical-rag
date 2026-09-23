import pytest

from src.embedding.simple import HashEmbeddingProvider
from src.indexer.qdrant_store import QdrantIndexStore
from src.schema import EvidenceItem


def items():
    return [
        EvidenceItem("a", "text", "book.pdf", "alpha provenance source", 1, 1),
        EvidenceItem("b", "text", "book.pdf", "beta image screenshot", 2, 2),
        EvidenceItem("c", "text", "book.pdf", "gamma offline mode", 3, 3),
    ]


def test_real_qdrant_hybrid_persistence_and_rebuild(tmp_path):
    provider = HashEmbeddingProvider()
    store = QdrantIndexStore(tmp_path / "index")
    try:
        store.build(items(), provider)
        for dense, sparse in ((3, 0), (0, 3), (3, 3)):
            hits = store.search("provenance", provider, dense_top_k=dense, bm25_top_k=sparse)
            assert hits[0].evidence.evidence_id == "a"
            assert hits[0].evidence.page_start == 1
        store.close()
        reopened = store.search("provenance", provider)
        assert reopened[0].evidence.source_file == "book.pdf"
        updated = list(reversed(items()))
        store.build(updated, provider)
        assert store.search("provenance", provider)[0].evidence.evidence_id == "a"
        store.build([items()[1]], provider)
        assert [h.evidence.evidence_id for h in store.search("image", provider)] == ["b"]
        store.build([], provider)
        assert store.search("anything", provider) == []
        assert not store.bm25_path.exists()
    finally:
        store.close()


def test_rebuild_with_different_embedding_dimensions(tmp_path):
    store = QdrantIndexStore(tmp_path / "index")
    try:
        store.build(items(), HashEmbeddingProvider(16))
        store.build(items(), HashEmbeddingProvider(32))
        assert store.search("provenance", HashEmbeddingProvider(32))[0].evidence.evidence_id == "a"
    finally:
        store.close()


def test_failed_embedding_does_not_clear_previous_index(tmp_path):
    store = QdrantIndexStore(tmp_path / "index")
    provider = HashEmbeddingProvider()

    class Broken:
        def embed_texts(self, texts):
            raise RuntimeError("embedding unavailable")

    try:
        store.build(items(), provider)
        with pytest.raises(RuntimeError):
            store.build(items(), Broken())
        assert store.search("provenance", provider)[0].evidence.evidence_id == "a"
    finally:
        store.close()
