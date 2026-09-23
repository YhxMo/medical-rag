"""Page-local overlapping chunks with stable source/page/offset identifiers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from src.schema import EvidenceItem, PageDocument


def chunk_pages(
    pages: Iterable[PageDocument], *, chunk_size=900, overlap=100
) -> list[EvidenceItem]:
    if not 0 <= overlap < chunk_size:
        raise ValueError("Require 0 <= overlap < chunk_size")
    evidence = []
    for page in pages:
        text = page.text.strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Prefer a paragraph, sentence or word boundary in the latter half.
            if end < len(text):
                lower = start + max(chunk_size // 2, overlap + 1)
                for delimiter in ("\n\n", "\n", ". ", "。", " "):
                    boundary = text.rfind(delimiter, lower, end)
                    if boundary >= 0:
                        end = boundary + len(delimiter)
                        break
            content = text[start:end].strip()
            if content:
                key = f"{page.source_file}:{page.page_number}:{start}:{content}"
                evidence.append(
                    EvidenceItem(
                        hashlib.sha256(key.encode()).hexdigest()[:24],
                        "text",
                        page.source_file,
                        content,
                        page.page_number,
                        page.page_number,
                        {"char_start": start, "char_end": end},
                    )
                )
            if end == len(text):
                break
            start = end - overlap
    return evidence
