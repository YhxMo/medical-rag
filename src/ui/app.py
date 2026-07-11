"""Gradio demo surface for source-grounded medical RAG retrieval."""
from __future__ import annotations

from collections.abc import Iterator
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.embedding.base import EmbeddingProvider
from src.indexer.base import EvidenceStore
from src.reranker.base import Reranker
from src.schema import RetrievalHit


LOGGER = logging.getLogger(__name__)
_EVIDENCE_HEADERS = ["排名", "最终分", "证据类型", "来源", "页码", "Dense", "BM25", "Rerank", "证据片段"]


class _DemoError(RuntimeError):
    """Expected demo error that is safe to render in the browser."""


@dataclass
class DemoController:
    """Run interactive retrieval without exposing configuration details to the UI."""

    store: EvidenceStore
    embedding_factory: Callable[[str], EmbeddingProvider]
    reranker_factory: Callable[[str], Reranker]
    generator_factory: Callable[[str], Any]
    _embeddings: dict[str, EmbeddingProvider] = field(default_factory=dict, init=False)
    _rerankers: dict[str, Reranker] = field(default_factory=dict, init=False)
    _generators: dict[str, Any] = field(default_factory=dict, init=False)

    def run(
        self,
        question: str,
        candidate_k: int,
        top_k: int,
        embedding_name: str,
        reranker_name: str,
        generator_name: str,
    ) -> tuple[str, str, list[list[str]], str]:
        try:
            normalized_question, hits, generator, candidate_count = self._prepare_request(
                question, candidate_k, top_k, embedding_name, reranker_name, generator_name
            )
            generated = generator.generate(normalized_question, hits)
        except _DemoError as exc:
            return _empty_result(str(exc))
        except Exception as exc:  # Keep operational details and local paths out of the browser.
            LOGGER.exception("Demo request failed: %s", type(exc).__name__)
            return _empty_result(_friendly_error(exc))

        answer = generated.answer.strip() or "未生成可展示的回答。"
        status = self._status(
            candidate_count, candidate_k, hits, top_k, embedding_name, reranker_name, generator_name
        )
        return answer, _format_sources(hits), _evidence_rows(hits), status

    def stream(
        self,
        question: str,
        candidate_k: int,
        top_k: int,
        embedding_name: str,
        reranker_name: str,
        generator_name: str,
    ) -> Iterator[tuple[str, str, list[list[str]], str]]:
        """Show retrieved evidence immediately, then update the answer per token."""
        try:
            normalized_question, hits, generator, candidate_count = self._prepare_request(
                question, candidate_k, top_k, embedding_name, reranker_name, generator_name
            )
            sources = _format_sources(hits)
            evidence_rows = _evidence_rows(hits)
            status = self._status(
                candidate_count, candidate_k, hits, top_k, embedding_name, reranker_name, generator_name
            )
            yield "", sources, evidence_rows, status + " | 正在生成回答..."

            stream = getattr(generator, "stream", None)
            if not callable(stream):
                generated = generator.generate(normalized_question, hits)
                answer = generated.answer.strip() or "未生成可展示的回答。"
                yield answer, sources, evidence_rows, status + " | 已完成"
                return

            answer_parts: list[str] = []
            for chunk in stream(normalized_question, hits):
                if not chunk:
                    continue
                answer_parts.append(str(chunk))
                yield "".join(answer_parts), sources, evidence_rows, status + " | 正在生成回答..."

            answer = "".join(answer_parts).strip() or "未生成可展示的回答。"
            yield answer, sources, evidence_rows, status + " | 已完成"
        except _DemoError as exc:
            yield _empty_result(str(exc))
        except Exception as exc:  # Keep operational details and local paths out of the browser.
            LOGGER.exception("Demo stream failed: %s", type(exc).__name__)
            yield _empty_result(_friendly_error(exc))

    def _prepare_request(
        self,
        question: str,
        candidate_k: int,
        top_k: int,
        embedding_name: str,
        reranker_name: str,
        generator_name: str,
    ) -> tuple[str, list[RetrievalHit], Any, int]:
        normalized_question = question.strip()
        if not normalized_question:
            raise _DemoError("请输入一个医学影像学问题。")
        if not self._index_is_available():
            raise _DemoError("索引尚未构建。请先完成索引构建后再启动演示。")

        candidate_limit = int(candidate_k)
        final_limit = int(top_k)
        if not 10 <= candidate_limit <= 50:
            raise _DemoError("Candidate K 需要设置在 10 到 50 之间。")
        if not 1 <= final_limit <= 10:
            raise _DemoError("Top K 需要设置在 1 到 10 之间。")

        embedding_provider = self._component(self._embeddings, embedding_name, self.embedding_factory)
        candidate_hits = self.store.search(
            normalized_question,
            embedding_provider,
            dense_top_k=candidate_limit,
            bm25_top_k=candidate_limit,
            final_top_k=candidate_limit,
        )
        if not candidate_hits:
            raise _DemoError("没有检索到可用证据。可以换一种问法或扩大 Candidate K。")

        reranker = self._component(self._rerankers, reranker_name, self.reranker_factory)
        hits = reranker.rerank(normalized_question, candidate_hits, top_k=final_limit)
        generator = self._component(self._generators, generator_name, self.generator_factory)
        self._validate_generator(generator_name, generator)
        return normalized_question, hits, generator, len(candidate_hits)

    @staticmethod
    def _status(
        candidate_count: int,
        candidate_k: int,
        hits: list[RetrievalHit],
        top_k: int,
        embedding_name: str,
        reranker_name: str,
        generator_name: str,
    ) -> str:
        return (
            f"候选 {candidate_count}/{int(candidate_k)} 条 -> 最终 {len(hits)}/{int(top_k)} 条 | "
            f"embedding: {embedding_name} | "
            f"reranker: {reranker_name} | generator: {generator_name}"
        )

    def _index_is_available(self) -> bool:
        evidence_path = getattr(self.store, "evidence_path", None)
        return evidence_path is None or Path(evidence_path).exists()

    @staticmethod
    def _component(cache: dict[str, Any], name: str, factory: Callable[[str], Any]) -> Any:
        if name not in cache:
            cache[name] = factory(name)
        return cache[name]

    @staticmethod
    def _validate_generator(name: str, generator: Any) -> None:
        if name == "dashscope" and not str(getattr(generator, "api_key", "")).strip():
            raise _DemoError("DashScope generator 未配置 API key。请选择 extractive，或在本地环境中配置密钥后重试。")


def create_app(
    store: EvidenceStore,
    *,
    embedding_factory: Callable[[str], EmbeddingProvider],
    reranker_factory: Callable[[str], Reranker],
    generator_factory: Callable[[str], Any],
    default_embedding: str = "bge",
    default_reranker: str = "none",
    default_generator: str = "extractive",
    default_candidate_k: int = 20,
    default_top_k: int = 5,
):
    """Create a compact local demo with answers, citations, and raw evidence."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("gradio is required to serve the UI.") from exc

    controller = DemoController(store, embedding_factory, reranker_factory, generator_factory)
    with gr.Blocks(title="医学影像学 RAG Demo") as demo:
        gr.Markdown("# 医学影像学 RAG Demo")

        question = gr.Textbox(
            label="问题",
            placeholder="例如：脑出血的 CT 表现是什么？",
            lines=3,
            max_lines=5,
        )
        with gr.Row(equal_height=True):
            candidate_k = gr.Slider(
                label="候选召回数 Candidate K",
                minimum=10,
                maximum=50,
                value=min(50, max(10, int(default_candidate_k))),
                step=1,
            )
            top_k = gr.Slider(
                label="最终证据数 Top K",
                minimum=1,
                maximum=10,
                value=min(10, max(1, int(default_top_k))),
                step=1,
            )

        with gr.Row(equal_height=True):
            embedding = gr.Dropdown(
                label="Embedding",
                choices=[("BGE", "bge"), ("Hash", "hash")],
                value=default_embedding,
            )
            reranker = gr.Dropdown(
                label="Reranker",
                choices=[("不重排", "none"), ("BGE Reranker", "bge")],
                value=default_reranker,
            )
            generator = gr.Dropdown(
                label="Generator",
                choices=[("Extractive", "extractive"), ("DashScope", "dashscope")],
                value=default_generator,
            )

        with gr.Row():
            submit = gr.Button("生成回答", variant="primary")
            clear = gr.Button("清空")

        status = gr.Markdown("等待提问", elem_id="status-output")
        gr.Markdown("### 回答")
        answer = gr.Markdown(elem_id="answer-output")
        gr.Markdown("### 引用来源")
        sources = gr.Markdown("暂无引用来源。")

        with gr.Accordion("检索证据", open=True):
            evidence = gr.Dataframe(
                headers=_EVIDENCE_HEADERS,
                datatype=["str"] * len(_EVIDENCE_HEADERS),
                value=[],
                interactive=False,
                wrap=True,
                max_height=320,
            )

        inputs = [question, candidate_k, top_k, embedding, reranker, generator]
        outputs = [answer, sources, evidence, status]
        submit.click(controller.stream, inputs=inputs, outputs=outputs)
        question.submit(controller.stream, inputs=inputs, outputs=outputs)
        clear.click(
            lambda: ("", "", "暂无引用来源。", [], "等待提问"),
            outputs=[question, answer, sources, evidence, status],
        )
    return demo


def _empty_result(message: str) -> tuple[str, str, list[list[str]], str]:
    return message, "暂无引用来源。", [], message


def _format_sources(hits: list[RetrievalHit]) -> str:
    lines = []
    for hit in hits:
        evidence = hit.evidence
        lines.append(
            f"- `{evidence.evidence_id}` | {_display_source(evidence.source_file)} | "
            f"{_page_label(evidence.page_start, evidence.page_end)} | `{evidence.evidence_type}` | "
            f"score {_format_score(hit.score)}"
        )
    return "\n".join(lines) if lines else "暂无引用来源。"


def _evidence_rows(hits: list[RetrievalHit]) -> list[list[str]]:
    rows = []
    for hit in hits:
        scores = hit.scores
        rows.append(
            [
                str(hit.rank),
                _format_score(hit.score),
                hit.evidence.evidence_type,
                _display_source(hit.evidence.source_file),
                _page_label(hit.evidence.page_start, hit.evidence.page_end),
                _format_score(scores.get("dense")),
                _format_score(scores.get("bm25")),
                _format_score(scores.get("rerank")),
                _excerpt(hit.evidence.content),
            ]
        )
    return rows


def _friendly_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "max_tokens" in message or ("invalidparameter" in message and "range" in message):
        return "DashScope 的 max_tokens 配置无效。请设置为 1 到 65536；本地 demo 建议使用 2048。"
    if isinstance(exc, FileNotFoundError) or "no such file" in message:
        return "索引或本地模型尚未准备完成。请先构建索引并确认模型可用。"
    if "api key" in message or "api_key" in message or "authentication" in message:
        return "模型服务认证未配置。请选择 extractive，或在本地环境中配置密钥后重试。"
    if "dimension" in message or "vector size" in message:
        return "当前 embedding 与索引不匹配。请选择构建该索引时使用的 embedding。"
    if "collection" in message or "index" in message or "evidence.jsonl" in message:
        return "索引尚不可用。请先完成索引构建后再查询。"
    if isinstance(exc, ImportError) or "not installed" in message:
        return "运行依赖未安装完成。请在本地环境安装项目依赖后重试。"
    return "本次检索未能完成。请检查索引与模型准备状态后重试。"


def _display_source(source_file: str) -> str:
    return Path(str(source_file).replace("\\", "/")).name or "未标注来源"


def _page_label(page_start: int | None, page_end: int | None) -> str:
    if page_start is None:
        return "未标注页码"
    if page_end is None or page_end == page_start:
        return f"p.{page_start}"
    return f"p.{page_start}-{page_end}"


def _format_score(value: float | None) -> str:
    return f"{float(value):.4f}" if value is not None else "-"


def _excerpt(content: str, *, limit: int = 220) -> str:
    normalized = " ".join(content.split())
    return normalized if len(normalized) <= limit else normalized[:limit].rstrip() + "..."
