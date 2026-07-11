"""Configurable textbook chunking strategies."""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from src.schema import PageDocument, TextChunk


_HEADING_PATTERN = re.compile(
    r"^\s*((第[一二三四五六七八九十百千万\d]+[章节篇])|([一二三四五六七八九十]+、)|([（(][一二三四五六七八九十]+[）)])|(\d+(?:\.\d+)*[\.、．]?\s+)).{0,80}$"
)


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "layout_heading"
    target_chars: int = 700
    max_chars: int = 1200
    min_chars: int = 120
    chunk_size: int = 512
    chunk_overlap: int = 50


class Chunker(ABC):
    """Base class for document chunking strategies."""

    @abstractmethod
    def chunk(self, pages: Iterable[PageDocument]) -> list[TextChunk]:
        """Split pages into chunks."""


def _chunk_id(source_file: str, strategy: str, page_start: int, content: str) -> str:
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    return f"{source_file}:{strategy}:p{page_start}:{digest}"


def _split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n|\r\n\s*\r\n", text) if part.strip()]


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and len(stripped) <= 90 and _HEADING_PATTERN.match(stripped))


def _heading_level(line: str) -> int:
    stripped = line.strip()
    if re.match(r"^第[一二三四五六七八九十百千万\d]+篇", stripped):
        return 1
    if re.match(r"^第[一二三四五六七八九十百千万\d]+章", stripped):
        return 2
    if re.match(r"^第[一二三四五六七八九十百千万\d]+节", stripped):
        return 3
    if re.match(r"^[一二三四五六七八九十]+、", stripped):
        return 4
    if re.match(r"^[（(][一二三四五六七八九十]+[）)]", stripped):
        return 5
    return 6


class LayoutHeadingChunker(Chunker):
    """Chunk pages while preserving headings, paragraphs, and page metadata."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, pages: Iterable[PageDocument]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        heading_by_level: dict[int, str] = {}
        buffer: list[str] = []
        page_start: int | None = None
        page_end: int | None = None
        needs_ocr = False

        def flush() -> None:
            nonlocal buffer, page_start, page_end, needs_ocr
            content = "\n\n".join(buffer).strip()
            if not content or page_start is None or page_end is None:
                buffer = []
                return
            chunks.append(
                TextChunk(
                    chunk_id=_chunk_id(chunks_source, self.config.strategy, page_start, content),
                    source_file=chunks_source,
                    page_start=page_start,
                    page_end=page_end,
                    heading_path=tuple(heading_by_level[level] for level in sorted(heading_by_level)),
                    content=content,
                    chunk_strategy=self.config.strategy,
                    needs_ocr=needs_ocr,
                )
            )
            buffer = []
            page_start = None
            page_end = None
            needs_ocr = False

        pages_list = list(pages)
        if not pages_list:
            return []
        chunks_source = pages_list[0].source_file

        for page in pages_list:
            paragraphs = _split_paragraphs(page.text) or ([page.text.strip()] if page.text.strip() else [])
            for paragraph in paragraphs:
                first_line = paragraph.splitlines()[0].strip()
                if _is_heading(first_line):
                    level = _heading_level(first_line)
                    if buffer and (
                        sum(len(item) for item in buffer) >= self.config.min_chars or level <= 4
                    ):
                        flush()
                    heading_by_level = {
                        existing_level: heading
                        for existing_level, heading in heading_by_level.items()
                        if existing_level < level
                    }
                    heading_by_level[level] = first_line
                    continue

                if page_start is None:
                    page_start = page.page_number
                page_end = page.page_number
                needs_ocr = needs_ocr or page.needs_ocr

                projected = sum(len(item) for item in buffer) + len(paragraph)
                if buffer and projected > self.config.max_chars:
                    flush()
                    page_start = page.page_number
                    page_end = page.page_number
                    needs_ocr = page.needs_ocr
                buffer.append(paragraph)

        flush()
        return chunks


class FixedChunker(Chunker):
    """Fixed-size baseline chunker for experiments."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig(strategy="fixed")

    def chunk(self, pages: Iterable[PageDocument]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for page in pages:
            text = page.text.strip()
            if not text:
                continue
            step = max(1, self.config.chunk_size - self.config.chunk_overlap)
            for start in range(0, len(text), step):
                content = text[start : start + self.config.chunk_size].strip()
                if not content:
                    continue
                chunks.append(
                    TextChunk(
                        chunk_id=_chunk_id(page.source_file, "fixed", page.page_number, content),
                        source_file=page.source_file,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        heading_path=(),
                        content=content,
                        chunk_strategy="fixed",
                        needs_ocr=page.needs_ocr,
                        metadata={"char_start": start},
                    )
                )
        return chunks


def build_chunker(strategy: str, config: dict | None = None) -> Chunker:
    values = dict(config or {})
    if strategy == "fixed":
        return FixedChunker(
            ChunkingConfig(
                strategy="fixed",
                chunk_size=int(values.get("chunk_size", 512)),
                chunk_overlap=int(values.get("chunk_overlap", 50)),
            )
        )
    if strategy in {"layout_heading", "parent_child", "semantic"}:
        # Parent-child and semantic use layout-aware behavior until their specialized
        # retrieval/indexing stages are implemented.
        return LayoutHeadingChunker(
            ChunkingConfig(
                strategy=strategy,
                target_chars=int(values.get("target_chars", 700)),
                max_chars=int(values.get("max_chars", 1200)),
                min_chars=int(values.get("min_chars", 120)),
            )
        )
    raise ValueError(f"Unknown chunking strategy: {strategy}")
