from types import SimpleNamespace

from src.generator.generator import DashScopeGenerator
from src.schema import EvidenceItem, RetrievalHit


class FakeCompletions:
    def create(self, **kwargs):
        assert kwargs["model"] == "qwen-plus"
        assert "证据" in kwargs["messages"][1]["content"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="脑出血CT可见高密度影。[1]"))]
        )


class FakeStreamingCompletions:
    def create(self, **kwargs):
        assert kwargs["stream"] is True
        return iter(
            [
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="脑出血CT"))]),
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]),
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="可见高密度影。[1]"))]),
            ]
        )


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


def test_dashscope_generator_builds_citation_prompt_with_fake_client():
    hit = RetrievalHit(EvidenceItem("e1", "text", "book.pdf", "脑出血CT可见高密度影。", 1, 1), 1.0, 1)
    generator = DashScopeGenerator(
        api_key="key",
        api_base="https://example.test/v1",
        model="qwen-plus",
        client=FakeClient(),
    )

    answer = generator.generate("脑出血CT表现？", [hit])

    assert "高密度影" in answer.answer
    assert answer.sources[0].evidence_id == "e1"


def test_dashscope_generator_streams_completion_deltas_with_fake_client():
    hit = RetrievalHit(EvidenceItem("e1", "text", "book.pdf", "脑出血CT可见高密度影。", 1, 1), 1.0, 1)
    generator = DashScopeGenerator(
        api_key="key",
        api_base="https://example.test/v1",
        model="qwen-plus",
        client=SimpleNamespace(chat=SimpleNamespace(completions=FakeStreamingCompletions())),
    )

    chunks = list(generator.stream("脑出血CT表现？", [hit]))

    assert chunks == ["脑出血CT", "可见高密度影。[1]"]
