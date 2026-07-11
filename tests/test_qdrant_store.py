from __future__ import annotations

from math import sqrt
from types import SimpleNamespace

import pytest

from src.config.settings import Settings
from src.embedding.simple import HashEmbeddingProvider
from src.indexer.factory import create_index_store
from src.indexer.qdrant_store import QdrantIndexStore
from src.indexer.store import IndexStore
from src.schema import EvidenceItem


class FakeModels:
    class Distance:
        COSINE = "cosine"

    class VectorParams:
        def __init__(self, *, size: int, distance: str) -> None:
            self.size = size
            self.distance = distance

    class PointStruct:
        def __init__(self, *, id: str, vector: list[float], payload: dict) -> None:
            self.id = id
            self.vector = vector
            self.payload = payload


class FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: dict[str, dict] = {}

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def delete_collection(self, *, collection_name: str) -> None:
        del self.collections[collection_name]

    def create_collection(self, *, collection_name: str, vectors_config) -> None:
        self.collections[collection_name] = {"vectors_config": vectors_config, "points": []}

    def upsert(self, *, collection_name: str, points: list, wait: bool) -> None:
        self.collections[collection_name]["points"] = points

    def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ):
        scored = [
            SimpleNamespace(payload=point.payload, score=_cosine(query, point.vector))
            for point in self.collections[collection_name]["points"]
        ]
        scored.sort(key=lambda point: point.score, reverse=True)
        return SimpleNamespace(points=scored[:limit])


def test_qdrant_store_persists_payload_and_runs_hybrid_retrieval(tmp_path):
    evidence = [
        EvidenceItem("e1", "text", "book.pdf", "脑出血 CT 高密度影", 1, 1, {"chapter": "脑"}),
        EvidenceItem("e2", "exercise_qa", "book.pdf", "肺炎 斑片状阴影", 2, 2),
    ]
    client = FakeQdrantClient()
    provider = HashEmbeddingProvider()
    store = QdrantIndexStore(
        tmp_path / "index",
        collection_name="medical_evidence",
        client=client,
        models_module=FakeModels,
    )

    store.build(evidence, provider)
    dense_hits = store.search("脑出血 CT", provider, dense_top_k=2, bm25_top_k=0, final_top_k=1)
    bm25_hits = store.search("脑出血 CT", provider, dense_top_k=0, bm25_top_k=2, final_top_k=1)

    assert dense_hits[0].evidence.evidence_id == "e1"
    assert dense_hits[0].scores["bm25"] == 0.0
    assert bm25_hits[0].evidence.evidence_id == "e1"
    assert bm25_hits[0].scores["dense"] == 0.0
    assert (tmp_path / "index" / "bm25.pkl").exists()
    assert (tmp_path / "index" / "evidence.jsonl").exists()

    point = client.collections["medical_evidence"]["points"][0]
    assert point.payload["evidence_id"] == "e1"
    assert point.payload["metadata"]["chapter"] == "脑"
    assert point.payload["_ordinal"] == 0
    assert client.collections["medical_evidence"]["vectors_config"].size == len(provider.embed_query("脑出血"))


def test_qdrant_store_can_disable_all_retrieval_and_create_temporary_store(tmp_path):
    evidence = [EvidenceItem("e1", "text", "book.pdf", "alpha target marker", 1, 1)]
    client = FakeQdrantClient()
    store = QdrantIndexStore(
        tmp_path / "index",
        collection_name="medical_evidence",
        client=client,
        models_module=FakeModels,
    )
    provider = HashEmbeddingProvider()
    store.build(evidence, provider)

    assert store.search("target", provider, dense_top_k=0, bm25_top_k=0, final_top_k=1) == []

    temporary = store.create_empty(tmp_path / "tmp" / "filtered")
    assert temporary.collection_name != store.collection_name
    assert temporary.index_dir == tmp_path / "tmp" / "filtered"


def test_qdrant_store_runs_with_embedded_client(tmp_path):
    pytest.importorskip("qdrant_client")
    provider = HashEmbeddingProvider()
    store = QdrantIndexStore(tmp_path / "index", local_path=tmp_path / "qdrant")
    store.build(
        [
            EvidenceItem("e1", "text", "book.pdf", "脑出血 CT 高密度影", 1, 1),
            EvidenceItem("e2", "text", "book.pdf", "肺炎 斑片状阴影", 2, 2),
        ],
        provider,
    )
    store.build([EvidenceItem("e1", "text", "book.pdf", "脑出血 CT 高密度影", 1, 1)], provider)

    hits = store.search("脑出血 CT", provider, dense_top_k=1, bm25_top_k=0, final_top_k=1)
    client, _ = store._connection()

    assert hits[0].evidence.evidence_id == "e1"
    assert client.count(collection_name=store.collection_name, exact=True).count == 1

    store.build([], provider)
    assert store.load_evidence() == []
    assert client.count(collection_name=store.collection_name, exact=True).count == 0
    client.close()


def test_store_factory_defaults_to_qdrant_and_keeps_faiss_opt_in(tmp_path):
    settings = Settings(
        path=tmp_path / "config.yaml",
        raw={
            "paths": {"artifact_dir": str(tmp_path / "artifacts"), "index_dir": str(tmp_path / "index")},
            "active": {"vector_store": "qdrant"},
        },
    )

    qdrant_store = create_index_store(settings)
    assert isinstance(qdrant_store, QdrantIndexStore)
    assert qdrant_store.local_path == tmp_path / "artifacts" / "qdrant"

    faiss_settings = Settings(path=tmp_path / "config.yaml", raw={"active": {"vector_store": "faiss"}})
    assert isinstance(create_index_store(faiss_settings), IndexStore)


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    denominator = sqrt(sum(a * a for a in left)) * sqrt(sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0
