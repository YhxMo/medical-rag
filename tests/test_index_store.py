from src.embedding.simple import HashEmbeddingProvider
from src.indexer.store import IndexStore
from src.schema import EvidenceItem


def test_index_store_builds_and_retrieves(tmp_path):
    evidence = [
        EvidenceItem("e1", "text", "book.pdf", "脑出血 CT 高密度影", 1, 1),
        EvidenceItem("e2", "text", "book.pdf", "肺炎 斑片状阴影", 2, 2),
    ]
    provider = HashEmbeddingProvider()
    store = IndexStore(tmp_path / "index")

    store.build(evidence, provider)
    hits = store.search("脑出血 CT", provider, final_top_k=1)

    assert hits[0].evidence.evidence_id == "e1"
    assert (tmp_path / "index" / "dense.faiss").exists()
    assert (tmp_path / "index" / "bm25.pkl").exists()
    assert (tmp_path / "index" / "evidence.jsonl").exists()


def test_index_store_can_disable_dense_or_bm25(tmp_path):
    evidence = [
        EvidenceItem("e1", "text", "book.pdf", "alpha target marker", 1, 1),
        EvidenceItem("e2", "text", "book.pdf", "gamma delta", 2, 2),
    ]
    provider = HashEmbeddingProvider()
    store = IndexStore(tmp_path / "index")
    store.build(evidence, provider)

    bm25_hits = store.search("target", provider, dense_top_k=0, bm25_top_k=2, final_top_k=1)
    dense_hits = store.search("target", provider, dense_top_k=2, bm25_top_k=0, final_top_k=1)
    disabled_hits = store.search("target", provider, dense_top_k=0, bm25_top_k=0, final_top_k=1)

    assert bm25_hits
    assert bm25_hits[0].scores["dense"] == 0.0
    assert dense_hits
    assert dense_hits[0].scores["bm25"] == 0.0
    assert disabled_hits == []


def test_index_store_applies_evidence_type_weights(tmp_path):
    evidence = [EvidenceItem("e1", "exercise_qa", "book.pdf", "alpha target marker", 1, 1)]
    provider = HashEmbeddingProvider()
    store = IndexStore(tmp_path / "index")
    store.build(evidence, provider)

    hit = store.search(
        "target",
        provider,
        dense_top_k=1,
        bm25_top_k=0,
        final_top_k=1,
        evidence_type_weights={"exercise_qa": 0.5},
    )[0]

    assert hit.scores["weighted"] == hit.scores["rrf"] * 0.5
