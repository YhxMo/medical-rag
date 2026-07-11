from __future__ import annotations

from pathlib import Path

from src.embedding.simple import HashEmbeddingProvider
from src.generator.generator import ExtractiveGenerator
from src.reranker.base import NoopReranker
from src.schema import EvidenceItem, RetrievalHit
from src.ui.app import DemoController, _friendly_error


class FakeStore:
    def __init__(self, hits: list[RetrievalHit], evidence_path: Path | None = None) -> None:
        self.hits = hits
        self.evidence_path = evidence_path
        self.last_search_kwargs = None

    def search(self, query, embedding_provider, *, dense_top_k, bm25_top_k, final_top_k):
        self.last_search_kwargs = {
            "dense_top_k": dense_top_k,
            "bm25_top_k": bm25_top_k,
            "final_top_k": final_top_k,
        }
        return self.hits[:final_top_k]


def test_demo_controller_returns_answer_sources_and_sanitized_evidence_rows():
    hit = RetrievalHit(
        evidence=EvidenceItem(
            "e1",
            "text",
            r"C:\private\medical\book.pdf",
            "脑出血 CT 通常表现为高密度影。",
            2,
            3,
        ),
        score=0.42,
        rank=1,
        scores={"dense": 0.9, "bm25": 4.0, "rerank": 0.7},
    )
    controller = DemoController(
        FakeStore([hit]),
        embedding_factory=lambda _: HashEmbeddingProvider(),
        reranker_factory=lambda _: NoopReranker(),
        generator_factory=lambda _: ExtractiveGenerator(),
    )

    answer, sources, rows, status = controller.run("脑出血 CT 表现", 20, 3, "hash", "none", "extractive")

    assert "来源" in answer
    assert "book.pdf" in sources
    assert "C:\\private" not in sources
    assert rows == [["1", "0.4200", "text", "book.pdf", "p.2-3", "0.9000", "4.0000", "0.7000", "脑出血 CT 通常表现为高密度影。"]]
    assert "候选 1/20 条" in status
    assert "最终 1/3 条" in status


def test_demo_controller_reports_missing_index_without_attempting_search(tmp_path):
    controller = DemoController(
        FakeStore([], evidence_path=tmp_path / "missing.jsonl"),
        embedding_factory=lambda _: HashEmbeddingProvider(),
        reranker_factory=lambda _: NoopReranker(),
        generator_factory=lambda _: ExtractiveGenerator(),
    )

    answer, sources, rows, status = controller.run("脑出血", 20, 3, "hash", "none", "extractive")

    assert "索引尚未构建" in answer
    assert sources == "暂无引用来源。"
    assert rows == []
    assert status == answer


def test_demo_controller_reports_missing_generator_api_key():
    hit = RetrievalHit(EvidenceItem("e1", "text", "book.pdf", "证据", 1, 1), 0.1, 1)

    class MissingKeyGenerator:
        api_key = ""

        def generate(self, question, hits):
            raise AssertionError("Generator should not run without a key.")

    controller = DemoController(
        FakeStore([hit]),
        embedding_factory=lambda _: HashEmbeddingProvider(),
        reranker_factory=lambda _: NoopReranker(),
        generator_factory=lambda _: MissingKeyGenerator(),
    )

    answer, _, rows, _ = controller.run("问题", 20, 3, "hash", "none", "dashscope")

    assert "API key" in answer
    assert rows == []


def test_demo_controller_streams_answer_after_showing_retrieved_evidence():
    hit = RetrievalHit(EvidenceItem("e1", "text", "book.pdf", "脑出血 CT 高密度影。", 1, 1), 0.1, 1)

    class StreamingGenerator:
        api_key = "key"

        def stream(self, question, hits):
            yield "脑出血 CT "
            yield "可见高密度影。[1]"

    controller = DemoController(
        FakeStore([hit]),
        embedding_factory=lambda _: HashEmbeddingProvider(),
        reranker_factory=lambda _: NoopReranker(),
        generator_factory=lambda _: StreamingGenerator(),
    )

    updates = list(controller.stream("脑出血 CT 表现", 20, 3, "hash", "none", "dashscope"))

    assert updates[0][0] == ""
    assert updates[0][2][0][3] == "book.pdf"
    assert "正在生成" in updates[0][3]
    assert updates[1][0] == "脑出血 CT "
    assert updates[2][0] == "脑出血 CT 可见高密度影。[1]"
    assert updates[-1][0] == "脑出血 CT 可见高密度影。[1]"
    assert updates[-1][3].endswith("已完成")


def test_demo_controller_retrieves_candidate_pool_before_final_top_k():
    hits = [
        RetrievalHit(EvidenceItem(f"e{index}", "text", "book.pdf", f"证据 {index}", index, index), 0.1, index)
        for index in range(1, 4)
    ]
    store = FakeStore(hits)

    class RecordingReranker:
        candidate_count = 0

        def rerank(self, query, hits, *, top_k=None):
            self.candidate_count = len(hits)
            return hits[:top_k]

    reranker = RecordingReranker()
    controller = DemoController(
        store,
        embedding_factory=lambda _: HashEmbeddingProvider(),
        reranker_factory=lambda _: reranker,
        generator_factory=lambda _: ExtractiveGenerator(),
    )

    _, _, rows, status = controller.run("问题", 20, 2, "hash", "bge", "extractive")

    assert store.last_search_kwargs == {"dense_top_k": 20, "bm25_top_k": 20, "final_top_k": 20}
    assert reranker.candidate_count == 3
    assert len(rows) == 2
    assert "候选 3/20 条" in status
    assert "最终 2/2 条" in status


def test_ui_maps_dashscope_max_tokens_error_to_actionable_message():
    message = _friendly_error(RuntimeError("InvalidParameter: max_tokens range"))

    assert "max_tokens" in message
    assert "2048" in message
