from types import SimpleNamespace as NS

import pytest
from PIL import Image

from src.application.client import ModelClient
from src.application.service import QueryService
from src.reranker.base import NoopReranker
from src.schema import EvidenceItem, RetrievalHit


def hit(content="source text", eid="a"):
    return RetrievalHit(EvidenceItem(eid, "text", "book.pdf", content, 3, 3), 1.0, 1)


class Store:
    def __init__(self, hits=None):
        self.hits = hits if hits is not None else [hit()]

    def search(self, query, embedding, **kwargs):
        self.query, self.options = query, kwargs
        return self.hits


class Model:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, system, payload, **kwargs):
        self.calls.append((system, payload, kwargs))
        return next(self.outputs)


def service(model=None, vision=None, **kwargs):
    return QueryService(Store(), None, NoopReranker(), model, vision, **kwargs)


def test_forward_answer_is_one_call_with_numbered_sources():
    model = Model("教材内容 [1]")
    result = service(model).ask("Explain this topic")
    assert result.answer == "教材内容 [1]" and not result.warnings
    assert len(model.calls) == 1
    assert model.calls[0][1]["sources"][0]["page"] == 3
    assert model.calls[0][1]["sources"][0]["id"] == 1
    assert "untrusted data" in model.calls[0][0]


def test_chinese_translation_then_answer():
    model = Model({"query": "resolution of 2 mm"}, "教材内容 [1]")
    result = service(model).ask("2毫米的分辨率")
    assert result.query == "resolution of 2 mm"
    assert len(model.calls) == 2
    assert model.calls[1][1]["question"] == "2毫米的分辨率"


def test_changed_translation_numbers_are_not_silently_used():
    with pytest.raises(ValueError, match="数字"):
        service(Model({"query": "20 mm"})).ask("2毫米")


def test_screenshot_read_then_answer_keeps_upload_separate(tmp_path):
    path = tmp_path / "upload.png"
    Image.new("RGB", (20, 20)).save(path)
    vision = Model({"query": "diagram relation"}, "图示关系 [1]")
    result = service(vision=vision).ask(image_path=path)
    assert result.query == "diagram relation"
    assert len(vision.calls) == 2
    assert vision.calls[1][2]["images"][0].startswith("data:image/png;base64,")
    assert len(result.sources) == 1 and result.sources[0].evidence.source_file == "book.pdf"
    assert path.exists()


def test_offline_never_calls_model_and_rejects_screenshot():
    model = Model()
    result = service(model).ask("教材问题", offline=True)
    assert "非模型生成答案" in result.answer and result.warnings
    assert not model.calls
    with pytest.raises(ValueError, match="视觉模型"):
        service(model).ask(image_path="unused.png", offline=True)


def test_invalid_citation_is_reported_without_a_repair_call():
    model = Model("Unsupported reference [9]")
    result = service(model).ask("a question")
    assert result.warnings and len(model.calls) == 1


def test_no_evidence_skips_generation():
    model = Model()
    app = QueryService(Store([]), None, NoopReranker(), model)
    assert "未检索到" in app.ask("question").answer
    assert not model.calls


def test_candidate_pool_is_reranked_before_context_cutoff():
    store = Store([hit("a", "a"), hit("b", "b"), hit("c", "c")])

    class Reverse:
        def rerank(self, query, hits, **kwargs):
            assert len(hits) == 3
            return list(reversed(hits))

    app = QueryService(store, None, Reverse(), candidate_k=3, top_k=1)
    result = app.ask("question", offline=True)
    assert result.sources[0].evidence.evidence_id == "c"
    assert store.options == {"dense_top_k": 3, "bm25_top_k": 3, "final_top_k": 3}


def test_context_deduplicates_and_truncates_without_mutating_index():
    store = Store([hit("short", "a"), hit("short", "a"), hit("long text here", "b")])
    app = QueryService(store, None, NoopReranker(), max_chars=10)
    result = app.ask("question", offline=True)
    assert [h.evidence.content for h in result.sources] == ["short", "long "]
    assert store.hits[2].evidence.content == "long text here"
    assert [h.rank for h in result.sources] == [1, 2]


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"candidate_k": 1}, {"max_chars": 0}])
def test_invalid_context_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        service(**kwargs)


def test_empty_question_fails_before_retrieval():
    with pytest.raises(ValueError, match="请输入"):
        service().ask(" ")


def test_model_client_request_and_incomplete_response():
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return NS(choices=[NS(finish_reason="stop", message=NS(content="answer [1]"))])

    sdk = NS(chat=NS(completions=NS(create=create)))
    model = ModelClient({"model": "fixture"}, client=sdk)
    assert model.generate("system", {"question": "q"}, images=["image-data"]) == "answer [1]"
    assert requests[0]["messages"][1]["content"][1]["image_url"]["url"] == "image-data"
    sdk.chat.completions.create = lambda **kw: NS(
        choices=[NS(finish_reason="length", message=NS(content="partial"))]
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        model.generate("system", "q")


def test_missing_credential_never_constructs_sdk():
    with pytest.raises(ValueError, match="API key"):
        ModelClient({}).generate("system", "question")
