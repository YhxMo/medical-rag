import json

from src import cli
from src.application.service import QueryService
from src.config.settings import load_settings
from src.embedding.simple import HashEmbeddingProvider
from src.indexer.qdrant_store import QdrantIndexStore
from src.reranker.base import NoopReranker
from src.schema import EvidenceItem, RetrievalHit
from src.ui.app import create_app, format_sources


def test_cli_and_ui_share_query_service(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    provider = HashEmbeddingProvider()
    store = QdrantIndexStore(tmp_path / "index")
    store.build(
        [EvidenceItem("a", "text", "book.pdf", "Source provenance evidence", 7, 7)], provider
    )
    service = QueryService(store, provider, NoopReranker())
    monkeypatch.setattr(cli, "build_service", lambda settings: service)
    assert cli.main(["query", "provenance", "--offline", "--json", "--config", str(config)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["sources"][0]["evidence"]["page_start"] == 7
    assert store._client is None  # CLI releases the local Qdrant lock.
    app = create_app(service)
    try:
        assert app.config["dependencies"]
    finally:
        app.close()


def test_cli_index_uses_same_configurable_store(tmp_path, monkeypatch, capsys):
    import pymupdf

    raw = tmp_path / "raw"
    raw.mkdir()
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Source provenance textbook content.")
        pdf.save(raw / "book.pdf")
    config = tmp_path / "config.yaml"
    config.write_text("paths:\n  data_dir: raw\n  artifact_dir: artifacts\n")
    monkeypatch.setattr(cli, "create_embedding", lambda settings: HashEmbeddingProvider())
    assert cli.main(["index", "--config", str(config)]) == 0
    assert "Indexed 1" in capsys.readouterr().out
    store = cli.create_store(load_settings(config))
    assert store.load_evidence()[0].source_file == "book.pdf"


def test_ui_escapes_source_content():
    evidence = EvidenceItem("a", "text", "<script>.pdf", "<img src=x onerror=alert(1)>", 1, 1)
    output = format_sources([RetrievalHit(evidence, 1, 1)])
    assert "<script>" not in output and "<img" not in output
    assert "&lt;script&gt;" in output


def test_cli_reports_missing_config(capsys):
    assert cli.main(["query", "q", "--config", "does-not-exist.yaml"]) == 1
    assert "Error:" in capsys.readouterr().err


def test_cli_evaluation_saves_tuning_inputs_and_per_question_results(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.yaml"
    config.write_text(
        "retrieval:\n  top_k: 1\nmodels:\n  text:\n    api_key: SECRET_NOT_FOR_REPORT\n"
    )
    dataset = tmp_path / "questions.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "provenance",
                    "expected_evidence_ids": ["a"],
                    "reviewed": True,
                }
            ]
        )
    )
    provider = HashEmbeddingProvider()
    store = QdrantIndexStore(tmp_path / "index")
    store.build([EvidenceItem("a", "text", "book.pdf", "Source provenance", 1, 1)], provider)
    service = QueryService(store, provider, NoopReranker(), top_k=1)
    monkeypatch.setattr(cli, "build_service", lambda settings: service)
    output = tmp_path / "run.json"
    assert (
        cli.main(["evaluate", str(dataset), "--config", str(config), "--output", str(output)]) == 0
    )
    raw = output.read_text()
    report = json.loads(raw)
    assert report["metrics"]["recall_at_k"] == 1
    assert report["details"][0]["question_id"] == "q1"
    assert len(report["inputs"]["evidence_sha256"]) == 64
    assert report["inputs"]["config"]["retrieval"]["top_k"] == 1
    assert "SECRET" not in raw and store._client is None
    assert "Saved evaluation record" in capsys.readouterr().out
