from src.config.settings import load_settings


def test_environment_and_paths_are_resolved_relative_to_config(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        "paths:\n  data_dir: inputs\nmodels:\n  text:\n    api_key: ${TEST_RAG_KEY}\n"
    )
    monkeypatch.setenv("TEST_RAG_KEY", "local-only")
    settings = load_settings(config)
    monkeypatch.chdir(tmp_path.parent)
    assert settings.get("models", "text", "api_key") == "local-only"
    assert settings.resolve_path("paths", "data_dir", default="data") == tmp_path / "inputs"
    assert (
        settings.resolve_path("paths", "artifact_dir", default="artifacts")
        == tmp_path / "artifacts"
    )
