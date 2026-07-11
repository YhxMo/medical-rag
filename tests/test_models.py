from src.schema import PageDocument


def test_page_document_marks_short_text_as_needing_ocr():
    page = PageDocument.from_text("book.pdf", 3, "太短", ocr_min_chars=50)

    assert page.text_length == 2
    assert page.needs_ocr is True


def test_page_document_keeps_normal_text_without_ocr_flag():
    text = "这是一个足够长的页面文本。" * 5
    page = PageDocument.from_text("book.pdf", 1, text, ocr_min_chars=50)

    assert page.text_length >= 50
    assert page.needs_ocr is False
