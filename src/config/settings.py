"""YAML configuration loading with environment variable resolution."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class Settings:
    """Project settings backed by a raw dictionary."""

    path: Path
    raw: dict[str, Any]

    def get(self, *keys: str, default: Any = None) -> Any:
        value: Any = self.raw
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    @property
    def active_chunking(self) -> str:
        return str(self.get("active", "chunking", default="layout_heading"))

    @property
    def ocr_min_chars(self) -> int:
        return int(self.get("vision_captioner", "ocr_text_min_chars", default=50))


def _resolve_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: os.getenv(match.group(1), ""), value)
    return value


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return Settings(path=config_path, raw=_resolve_env(raw))
