from pathlib import Path

from src.config.settings import load_settings


def test_load_settings_resolves_environment(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
active:
  chunking: layout_heading
vision_captioner:
  ocr_text_min_chars: 50
llm:
  api_key: "${DASHSCOPE_API_KEY}"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret")

    settings = load_settings(config)

    assert settings.active_chunking == "layout_heading"
    assert settings.ocr_min_chars == 50
    assert settings.get("llm", "api_key") == "secret"
    assert isinstance(settings.path, Path)
