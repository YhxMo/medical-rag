from src.schema import PageDocument


def test_page_document_loader_rule_under_50_chars_needs_ocr():
    page = PageDocument.from_text("scan.pdf", 1, "只有几个字", ocr_min_chars=50)

    assert page.needs_ocr
    assert page.text_length < 50
