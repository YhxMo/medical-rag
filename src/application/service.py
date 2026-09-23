"""Shared query flow for the CLI and web UI."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field, replace

from src.application.images import prepare_image
from src.application.prompts import ANSWER, SCREENSHOT, TRANSLATE
from src.schema import RetrievalHit


@dataclass
class QueryResult:
    answer: str
    query: str
    sources: list[RetrievalHit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def pack_context(hits, top_k=5, max_chars=6000):
    """Deduplicate and cap context; retain oversized hits as explicit excerpts."""
    chosen, seen, remaining = [], set(), max_chars
    for hit in hits:
        item = hit.evidence
        if item.evidence_id in seen or not item.content.strip():
            continue
        seen.add(item.evidence_id)
        excerpt = replace(item, content=item.content[:remaining])
        chosen.append(replace(hit, evidence=excerpt, rank=len(chosen) + 1))
        remaining -= len(excerpt.content)
        if len(chosen) == top_k or remaining == 0:
            break
    return chosen


class QueryService:
    def __init__(
        self,
        store,
        embedding,
        reranker,
        text_client=None,
        vision_client=None,
        *,
        candidate_k=20,
        top_k=5,
        max_chars=6000,
        translate_queries=True,
    ):
        if not 1 <= top_k <= candidate_k or max_chars < 1:
            raise ValueError("Require 1 <= top_k <= candidate_k and max_chars > 0")
        self.store, self.embedding, self.reranker = store, embedding, reranker
        self.text, self.vision = text_client, vision_client
        self.candidate_k, self.top_k, self.max_chars = candidate_k, top_k, max_chars
        self.translate_queries = translate_queries

    def retrieve(self, query):
        hits = self.store.search(
            query,
            self.embedding,
            dense_top_k=self.candidate_k,
            bm25_top_k=self.candidate_k,
            final_top_k=self.candidate_k,
        )
        hits = self.reranker.rerank(query, hits, top_k=self.candidate_k)
        return pack_context(hits, self.top_k, self.max_chars)

    def ask(self, question="", image_path=None, *, offline=False) -> QueryResult:
        start = time.perf_counter()
        question = question.strip()
        if not question and not image_path:
            raise ValueError("请输入问题或上传教材截图。")
        images, warnings = [], []
        query = question
        if image_path:
            if offline or self.vision is None:
                raise ValueError("截图提问需要启用并配置视觉模型。")
            question = question or "请解释截图中的内容，并提供教材依据。"
            images = [prepare_image(image_path)]
            query = self._query(
                self.vision.generate(
                    SCREENSHOT, {"question": question}, images=images, json_output=True
                )
            )
        elif self.translate_queries and re.search(r"[\u4e00-\u9fff]", question):
            if offline:
                warnings.append("离线模式使用原问题检索；英文教材建议用英文提问。")
            else:
                if self.text is None:
                    raise ValueError("请配置文本模型，或使用离线检索。")
                query = self._query(
                    self.text.generate(TRANSLATE, {"question": question}, json_output=True)
                )
                if set(re.findall(r"\d+(?:\.\d+)?", query)) != set(
                    re.findall(r"\d+(?:\.\d+)?", question)
                ):
                    raise ValueError("检索翻译改变了问题中的数字，请改用英文提问。")
        sources = self.retrieve(query)
        if not sources:
            answer = "未检索到教材证据。请先建立索引，或调整问题。"
        elif offline:
            answer = "以下为本地检索片段，非模型生成答案：\n\n" + "\n\n".join(
                f"[{i}] {hit.evidence.content}" for i, hit in enumerate(sources, 1)
            )
        else:
            client = self.vision if images else self.text
            if client is None:
                raise ValueError("请配置文本模型，或使用离线检索。")
            answer = client.generate(
                ANSWER,
                {
                    "question": question,
                    "sources": [
                        {
                            "id": i,
                            "source": h.evidence.source_file,
                            "page": h.evidence.page_start,
                            "type": h.evidence.evidence_type,
                            "content": h.evidence.content,
                        }
                        for i, h in enumerate(sources, 1)
                    ],
                },
                images=images,
            )
            citations = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
            if not citations or any(n < 1 or n > len(sources) for n in citations):
                warnings.append("回答缺少引用或包含无效编号，请核对下方原文。")
        return QueryResult(answer, query, sources, warnings, time.perf_counter() - start)

    @staticmethod
    def _query(output):
        query = output.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise ValueError("模型未返回有效的检索问题。")
        return query.strip()
