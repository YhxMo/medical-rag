from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from src.document.chunker import chunk_pages
from src.document.ingest import ingest
from src.document.loader import load_pdf
from src.embedding.simple import HashEmbeddingProvider
from src.indexer.qdrant_store import QdrantIndexStore
from src.schema import PageDocument


def test_chunking_covers_text_with_bounded_overlap_and_page_provenance():
    text = "1. Chapter heading\n\n" + "A sentence about source provenance. " * 30
    pages = [PageDocument.from_text("book.pdf", n, text) for n in (1, 2)]
    chunks = chunk_pages(pages, chunk_size=200, overlap=20)
    assert all(len(c.content) <= 200 and c.page_start == c.page_end for c in chunks)
    assert len({c.evidence_id for c in chunks}) == len(chunks)
    assert chunks[0].content.startswith("1. Chapter heading")
    covered = set()
    for chunk in [c for c in chunks if c.page_start == 1]:
        covered.update(range(chunk.metadata["char_start"], chunk.metadata["char_end"]))
    assert covered == set(range(len(text.strip())))
    assert chunks == chunk_pages(pages, chunk_size=200, overlap=20)


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (10, -1)])
def test_invalid_chunking_config(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=size, overlap=overlap)


def test_pdf_ingestion_to_real_index_and_optional_caption(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pdf_path = raw / "textbook.pdf"
    picture = tmp_path / "picture.png"
    Image.new("RGB", (50, 50), "blue").save(picture)
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text(
            (72, 72), "Source provenance records the textbook title and PDF page number."
        )
        page.insert_image(pymupdf.Rect(72, 120, 200, 200), filename=str(picture))
        pdf.save(pdf_path)
    (raw / "._textbook.pdf").write_text("macOS sidecar, not a PDF")
    assert load_pdf(pdf_path)[0].page_number == 1

    class Vision:
        available = True

        def generate(self, system, payload, **kwargs):
            assert "provenance" in payload["nearby_text"]
            assert kwargs["images"][0].startswith("data:image/png;base64,")
            return "A blue square used as a demonstration figure."

    provider = HashEmbeddingProvider()
    store = QdrantIndexStore(tmp_path / "index")
    try:
        assert ingest(raw, store, provider) == 1
        assert store.search("provenance", provider)[0].evidence.page_start == 1
        assert ingest(raw, store, provider, vision=Vision()) == 2
        caption = next(e for e in store.load_evidence() if e.evidence_type == "image_caption")
        assert caption.metadata["derived_by_model"]
        assert Path(caption.metadata["image_path"]).exists()
    finally:
        store.close()


def test_empty_folder_does_not_replace_index(tmp_path):
    with pytest.raises(ValueError, match="No PDF"):
        ingest(tmp_path, None, None)
