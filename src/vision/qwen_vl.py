"""Local Qwen-VL captioner.

This module loads the model lazily so tests and CLI help do not initialize a VLM.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.schema import ImageAsset, ImageCaption
from src.vision.base import VisionCaptioner


@dataclass
class QwenVLCaptioner(VisionCaptioner):
    """Caption images with a local Qwen-VL model through transformers."""

    model_name: str
    device_map: str = "auto"
    torch_dtype: str = "auto"

    def __post_init__(self) -> None:
        self._pipeline = None

    def caption(self, image: ImageAsset, *, prompt: str | None = None, is_ocr_text: bool = False) -> ImageCaption:
        pipe = self._load_pipeline()
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": str(image.path)},
                    {"type": "text", "text": prompt or "请用中文描述这张医学教材图片。"},
                ],
            }
        ]
        result = pipe(text=message, max_new_tokens=512)
        caption = _extract_text(result)
        return ImageCaption(
            caption_id=f"cap_{image.image_id}",
            image_id=image.image_id,
            source_file=image.source_file,
            page_number=image.page_number,
            caption=caption,
            is_ocr_text=is_ocr_text,
            metadata={"model": self.model_name, "prompt": prompt or ""},
        )

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("transformers is required for local Qwen-VL captioning.") from exc

        self._pipeline = pipeline(
            "image-text-to-text",
            model=self.model_name,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
        )
        return self._pipeline


def _extract_text(result) -> str:
    if isinstance(result, list) and result:
        item = result[0]
        if isinstance(item, dict):
            generated = item.get("generated_text")
            if isinstance(generated, str):
                return generated.strip()
            if isinstance(generated, list) and generated:
                last = generated[-1]
                if isinstance(last, dict):
                    content = last.get("content")
                    if isinstance(content, str):
                        return content.strip()
    return str(result).strip()
