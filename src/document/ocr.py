"""Optional OCR fallback for scanned PDF pages."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from src.document.loader import PDFLoader
from src.schema import PageDocument


@dataclass(frozen=True)
class OCRConfig:
    """Runtime settings for page-level OCR."""

    cache_dir: Path
    scale: float = 1.0
    min_confidence: float = 0.5
    min_chars: int = 50
    use_cls: bool = False


class RapidOCRPDFLoader:
    """Load PDF text and OCR pages that do not contain embedded text.

    OCR output is cached per source page because full textbook scans are slow to
    process. The cache stores plain text plus simple metadata so indexing can be
    repeated without re-running OCR.
    """

    def __init__(self, file_path: str | Path, config: OCRConfig) -> None:
        self.file_path = Path(file_path)
        self.config = config

    def load(self) -> list[PageDocument]:
        pages = PDFLoader(self.file_path, ocr_min_chars=self.config.min_chars).load()
        ocr_targets = [page for page in pages if page.needs_ocr]
        if not ocr_targets:
            return pages

        try:
            import fitz
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "rapidocr-onnxruntime and PyMuPDF are required for OCR fallback."
            ) from exc

        ocr = RapidOCR()
        page_by_number = {page.page_number: page for page in pages}
        with fitz.open(self.file_path) as pdf:
            for page in ocr_targets:
                cached = self._load_cached(page.page_number)
                if cached is None:
                    image = _render_page(pdf[page.page_number - 1], self.config.scale)
                    rows, elapsed = ocr(image, use_cls=self.config.use_cls)
                    text = _rows_to_text(
                        rows or [],
                        width=image.width,
                        height=image.height,
                        min_confidence=self.config.min_confidence,
                    )
                    cached = {"text": text, "elapsed": elapsed}
                    self._save_cached(page.page_number, cached)

                if len(cached.get("text", "").strip()) >= len(page.text.strip()):
                    page_by_number[page.page_number] = PageDocument.from_text(
                        source_file=page.source_file,
                        page_number=page.page_number,
                        text=cached.get("text", ""),
                        ocr_min_chars=self.config.min_chars,
                        metadata={
                            **page.metadata,
                            "ocr_provider": "rapidocr_onnxruntime",
                            "ocr_cache": str(self._cache_path(page.page_number)),
                        },
                    )

        return [page_by_number[index] for index in sorted(page_by_number)]

    def _cache_path(self, page_number: int) -> Path:
        stem_dir = self.config.cache_dir / self.file_path.stem
        scale_tag = str(self.config.scale).replace(".", "_")
        return stem_dir / f"p{page_number:04d}_s{scale_tag}.json"

    def _load_cached(self, page_number: int) -> dict | None:
        path = self._cache_path(page_number)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_cached(self, page_number: int, payload: dict) -> None:
        path = self._cache_path(page_number)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


def _render_page(page, scale: float) -> Image.Image:
    import fitz

    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def _rows_to_text(rows: Iterable, *, width: int, height: int, min_confidence: float) -> str:
    normalized = []
    for row in rows:
        if len(row) < 3:
            continue
        box, text, confidence = row[0], str(row[1]).strip(), float(row[2])
        if not text or confidence < min_confidence:
            continue
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        if _is_margin_noise(text, left=left, right=right, top=top, bottom=bottom, width=width, height=height):
            continue
        normalized.append((top, left, text))

    normalized.sort(key=lambda item: (round(item[0] / 8), item[1]))
    return "\n".join(text for _, _, text in normalized)


def _is_margin_noise(
    text: str,
    *,
    left: float,
    right: float,
    top: float,
    bottom: float,
    width: int,
    height: int,
) -> bool:
    box_width = right - left
    if right < width * 0.12 and box_width < width * 0.05 and len(text) <= 4:
        return True
    if text.isdigit() and (bottom < height * 0.08 or top > height * 0.92):
        return True
    return False
