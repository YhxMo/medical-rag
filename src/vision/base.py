"""Vision captioning interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.schema import ImageAsset, ImageCaption


class VisionCaptioner(ABC):
    """Generate captions or OCR-style text from images."""

    @abstractmethod
    def caption(self, image: ImageAsset, *, prompt: str | None = None, is_ocr_text: bool = False) -> ImageCaption:
        """Caption a single image."""

    def caption_many(
        self,
        images: list[ImageAsset],
        *,
        prompt: str | None = None,
        is_ocr_text: bool = False,
    ) -> list[ImageCaption]:
        return [self.caption(image, prompt=prompt, is_ocr_text=is_ocr_text) for image in images]
