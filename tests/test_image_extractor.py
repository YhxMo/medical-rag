from pathlib import Path

import fitz

from src.document.image_extractor import ImageExtractionConfig, PDFImageExtractor


def test_pdf_image_extractor_renders_page_image(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "医学影像学测试页面")
    doc.save(pdf_path)
    doc.close()

    extractor = PDFImageExtractor(
        ImageExtractionConfig(output_dir=tmp_path / "images", extract_embedded_images=False)
    )
    assets = extractor.extract(pdf_path)

    assert len(assets) == 1
    assert assets[0].kind == "page"
    assert assets[0].path.exists()
    assert assets[0].source_file == "sample.pdf"
    assert isinstance(assets[0].path, Path)
