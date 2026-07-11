"""Answer generation interfaces and generators."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from src.retrieval.base import cite
from src.schema import GeneratedAnswer, RetrievalHit


class ExtractiveGenerator:
    """Simple citation-preserving generator for tests and offline smoke runs."""

    def generate(self, question: str, hits: list[RetrievalHit]) -> GeneratedAnswer:
        if not hits:
            return GeneratedAnswer(answer="未检索到足够证据回答该问题。", sources=())

        lines = [f"问题：{question}", "", "基于当前检索证据，可参考："]
        for hit in hits:
            snippet = hit.evidence.content.replace("\n", " ")[:180]
            lines.append(f"- {snippet}（来源：{cite(hit.evidence)}）")
        return GeneratedAnswer(
            answer="\n".join(lines),
            sources=tuple(hit.evidence for hit in hits),
            metadata={"generator": "extractive"},
        )

    def stream(self, question: str, hits: list[RetrievalHit]) -> Iterator[str]:
        """Provide the same interface as remote generators for the demo UI."""
        yield self.generate(question, hits).answer


@dataclass
class DashScopeGenerator:
    """OpenAI-compatible DashScope chat generator."""

    api_key: str
    api_base: str
    model: str = "qwen-plus"
    temperature: float = 0.1
    max_tokens: int = 2048
    client: object | None = None

    def generate(self, question: str, hits: list[RetrievalHit]) -> GeneratedAnswer:
        if not hits:
            return GeneratedAnswer(answer="未检索到足够证据回答该问题。", sources=())

        client = self.client or self._build_client()
        prompt = _build_citation_prompt(question, hits)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "你是医学影像学学习助手。只能依据给定证据回答，并必须给出来源引用。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        answer = response.choices[0].message.content
        return GeneratedAnswer(
            answer=answer,
            sources=tuple(hit.evidence for hit in hits),
            metadata={"generator": "dashscope", "model": self.model},
        )

    def stream(self, question: str, hits: list[RetrievalHit]) -> Iterator[str]:
        """Yield DashScope completion deltas for interactive clients."""
        if not hits:
            yield "未检索到足够证据回答该问题。"
            return

        client = self.client or self._build_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "你是医学影像学学习助手。只能依据给定证据回答，并必须给出来源引用。",
                },
                {"role": "user", "content": _build_citation_prompt(question, hits)},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in response:
            choices = getattr(chunk, "choices", ())
            if not choices:
                continue
            content = getattr(getattr(choices[0], "delta", None), "content", None)
            if content:
                yield str(content)

    def _build_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai is required for DashScope generation.") from exc
        return OpenAI(api_key=self.api_key, base_url=self.api_base)


def _build_citation_prompt(question: str, hits: list[RetrievalHit]) -> str:
    evidence_blocks = []
    for index, hit in enumerate(hits, start=1):
        evidence_blocks.append(
            f"[{index}] 来源：{cite(hit.evidence)}\n"
            f"证据类型：{hit.evidence.evidence_type}\n"
            f"内容：{hit.evidence.content}"
        )
    return (
        f"问题：{question}\n\n"
        "请基于以下证据回答。要求：\n"
        "1. 用中文回答。\n"
        "2. 不要编造证据中没有的信息。\n"
        "3. 每个关键结论后标注来源编号，如 [1]。\n\n"
        "证据：\n" + "\n\n".join(evidence_blocks)
    )
