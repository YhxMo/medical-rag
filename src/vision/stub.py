"""Deterministic vision captioner for tests and dry runs."""
from __future__ import annotations

from src.schema import ImageAsset, ImageCaption
from src.vision.base import VisionCaptioner


class StubVisionCaptioner(VisionCaptioner):
    """Return stable captions without loading a model."""

    def caption(self, image: ImageAsset, *, prompt: str | None = None, is_ocr_text: bool = False) -> ImageCaption:
        prefix = "OCR" if is_ocr_text else "Caption"
        return ImageCaption(
            caption_id=f"cap_{image.image_id}",
            image_id=image.image_id,
            source_file=image.source_file,
            page_number=image.page_number,
            caption=f"{prefix}: {image.source_file} 第{image.page_number}页 {image.kind} image.",
            is_ocr_text=is_ocr_text,
            metadata={"prompt": prompt or ""},
        )
