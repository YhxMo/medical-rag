from pathlib import Path

from src.schema import ImageAsset
from src.vision.stub import StubVisionCaptioner


def test_stub_captioner_marks_ocr_caption():
    image = ImageAsset("img1", "book.pdf", 3, Path("page.png"), kind="page")

    caption = StubVisionCaptioner().caption(image, prompt="OCR prompt", is_ocr_text=True)

    assert caption.caption_id == "cap_img1"
    assert caption.is_ocr_text is True
    assert "第3页" in caption.caption
