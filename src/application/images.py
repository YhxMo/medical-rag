"""Bounded image decoding; never serve arbitrary filesystem paths."""

from __future__ import annotations
import base64
import hashlib
import io
import json
from pathlib import Path
from PIL import Image, ImageOps


def prepare_image(path: str | Path) -> tuple[str, str]:
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
        original.load()
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((1600, 1600))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    raw = buffer.getvalue()
    return "data:image/png;base64," + base64.b64encode(raw).decode(), hashlib.sha256(
        raw
    ).hexdigest()


def registered_image(path: str, allowed: dict[str, str]) -> Path:
    resolved = Path(path).resolve()
    expected = allowed.get(str(resolved))
    if (
        expected is None
        or hashlib.sha256(resolved.read_bytes()).hexdigest() != expected
    ):
        raise ValueError("Image is absent from the frozen source registry or changed")
    return resolved


def normalize_visual_fields(raw, fields):
    """Accept text containers without inventing missing or non-text observations.

    Structured text keeps its labels and nesting as JSON; the untouched response
    remains in the API cache for source review.
    """

    def text_container(value, depth=0):
        if depth > 8:
            return False
        if isinstance(value, str):
            return True
        if isinstance(value, list):
            return all(text_container(item, depth + 1) for item in value)
        if isinstance(value, dict):
            return all(
                isinstance(key, str) and text_container(item, depth + 1)
                for key, item in value.items()
            )
        return False

    if not isinstance(raw, dict):
        raise ValueError("Visual response must be a JSON object")
    normalized, conversions = {}, []
    for name in fields:
        value = raw.get(name)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            value = "\n".join(value)
            conversions.append(name)
        elif isinstance(value, (dict, list)) and text_container(value):
            value = json.dumps(value, ensure_ascii=False)
            conversions.append(name)
        if not isinstance(value, str):
            raise ValueError(f"Invalid visual field: {name}")
        normalized[name] = value
    return normalized, conversions
