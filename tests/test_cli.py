from src.cli import (
    _build_captioner,
    _build_generator,
    _build_reranker,
    _resolve_embedding_name,
    _resolve_reranker_name,
    main,
)
from src.config.settings import load_settings
from src.generator.generator import DashScopeGenerator, ExtractiveGenerator
from src.reranker.base import NoopReranker
from src.reranker.bge import BGEReranker
from src.vision.openai_vision import OpenAIVisionCaptioner


def test_cli_help_runs(capsys):
    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "medical-rag" in captured.out


def test_build_reranker_none_returns_noop(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("active:\n  reranker: none\n", encoding="utf-8")
    settings = load_settings(config)

    reranker = _build_reranker("none", settings)

    assert isinstance(reranker, NoopReranker)


def test_active_embedding_sets_cli_default(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("active:\n  embedding: bge_small_zh\n", encoding="utf-8")
    settings = load_settings(config)

    assert _resolve_embedding_name(None, settings) == "bge"


def test_explicit_embedding_overrides_active_default(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("active:\n  embedding: bge_small_zh\n", encoding="utf-8")
    settings = load_settings(config)

    assert _resolve_embedding_name("hash", settings) == "hash"


def test_active_reranker_sets_cli_default(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("active:\n  reranker: bge_base\n", encoding="utf-8")
    settings = load_settings(config)

    assert _resolve_reranker_name(None, settings) == "bge"


def test_build_reranker_bge_reads_config_model_name(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
reranker:
  bge_base:
    model: "local/reranker-model"
    device: "cpu"
""",
        encoding="utf-8",
    )
    settings = load_settings(config)

    reranker = _build_reranker("bge", settings)

    assert isinstance(reranker, BGEReranker)
    assert reranker.model_name == "local/reranker-model"
    assert reranker.device == "cpu"


def test_build_generator_extractive_returns_extractive_generator(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("active:\n  chunking: layout_heading\n", encoding="utf-8")
    settings = load_settings(config)

    generator = _build_generator("extractive", settings)

    assert isinstance(generator, ExtractiveGenerator)


def test_build_generator_dashscope_reads_llm_config(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
llm:
  api_base: "https://example.test/v1"
  api_key: "${TEST_DASHSCOPE_KEY}"
  model: "qwen3.6-plus"
  temperature: 0.2
  max_tokens: 4096
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_DASHSCOPE_KEY", "secret-key")
    settings = load_settings(config)

    generator = _build_generator("dashscope", settings)

    assert isinstance(generator, DashScopeGenerator)
    assert generator.api_base == "https://example.test/v1"
    assert generator.api_key == "secret-key"
    assert generator.model == "qwen3.6-plus"
    assert generator.temperature == 0.2
    assert generator.max_tokens == 4096


def test_build_captioner_dashscope_reads_vision_or_llm_config(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
llm:
  api_base: "https://llm.example/v1"
  api_key: "${TEST_DASHSCOPE_KEY}"
vision_captioner:
  api_model: "qwen-vl-plus"
  temperature: 0.3
  max_tokens: 700
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_DASHSCOPE_KEY", "secret-key")
    settings = load_settings(config)

    captioner = _build_captioner("dashscope", settings)

    assert isinstance(captioner, OpenAIVisionCaptioner)
    assert captioner.api_base == "https://llm.example/v1"
    assert captioner.api_key == "secret-key"
    assert captioner.model == "qwen-vl-plus"
    assert captioner.temperature == 0.3
    assert captioner.max_tokens == 700
