import fitz

from src.cli import _load_evidence
from src.config.settings import load_settings


def test_load_evidence_adds_contextual_image_caption_for_page_images(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pdf_path = data_dir / "book.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Chapter 1\n\nFigure 1-1 Head CT\nHemorrhage is high density on CT.")
    doc.save(pdf_path)
    doc.close()

    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
paths:
  data_dir: "{data_dir.as_posix()}"
  artifact_dir: "{(tmp_path / 'artifacts').as_posix()}"
  index_dir: "{(tmp_path / 'artifacts' / 'index').as_posix()}"
active:
  chunking: "layout_heading"
chunking:
  layout_heading:
    target_chars: 700
    max_chars: 1200
    min_chars: 10
image_caption_context:
  max_chars: 300
  max_chunks: 2
""",
        encoding="utf-8",
    )
    settings = load_settings(config)

    evidence = _load_evidence(settings, image_mode="all-pages", captioner_name="stub", ocr_mode="never")

    image_evidence = [item for item in evidence if item.evidence_type == "image_caption"]
    assert len(image_evidence) == 1
    assert image_evidence[0].metadata["image_kind"] == "page"
    assert "Figure 1-1 Head CT" in image_evidence[0].content
    assert "Hemorrhage is high density on CT" in image_evidence[0].content
