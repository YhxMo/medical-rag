"""Validate and resize user screenshots before sending them to the vision model."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageOps


def prepare_image(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("Only PNG, JPEG and WebP images are supported")
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Image exceeds 10 MB")
    with Image.open(path) as original:
        if original.format not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError("Unsupported image encoding")
        if original.width * original.height > 20_000_000:
            raise ValueError("Image exceeds 20 million pixels")
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((1600, 1600))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
