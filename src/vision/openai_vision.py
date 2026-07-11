"""OpenAI-compatible vision captioner for DashScope-style VLM APIs."""
from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass

from src.schema import ImageAsset, ImageCaption
from src.vision.base import VisionCaptioner


@dataclass
class OpenAIVisionCaptioner(VisionCaptioner):
    """Caption images through an OpenAI-compatible chat completions API."""

    api_key: str
    api_base: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 512
    client: object | None = None

    def caption(self, image: ImageAsset, *, prompt: str | None = None, is_ocr_text: bool = False) -> ImageCaption:
        # Build the OpenAI client lazily on first call and reuse it. The
        # OpenAI SDK client is thread-safe (its underlying httpx.Client
        # uses a connection pool), so reusing one client across a
        # ThreadPoolExecutor keeps HTTP keep-alive connections alive across
        # worker threads and avoids per-call TLS handshake overhead.
        client = self.client or self._build_client()
        self.client = client
        prompt_text = prompt or "请用中文描述这张医学教材图片。"
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                    ],
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        caption = response.choices[0].message.content
        return ImageCaption(
            caption_id=f"cap_{image.image_id}",
            image_id=image.image_id,
            source_file=image.source_file,
            page_number=image.page_number,
            caption=caption.strip(),
            is_ocr_text=is_ocr_text,
            metadata={"model": self.model, "prompt": prompt_text, "provider": "openai_vision"},
        )

    def _build_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai is required for OpenAI-compatible vision captioning.") from exc
        return OpenAI(api_key=self.api_key, base_url=self.api_base)


def _image_data_url(image: ImageAsset) -> str:
    mime_type, _ = mimetypes.guess_type(str(image.path))
    mime_type = mime_type or "image/png"
    payload = base64.b64encode(image.path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"
