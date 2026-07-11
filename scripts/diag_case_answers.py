"""Inspect case-analysis ANSWER blocks to see how answers are numbered."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.document.ocr import OCRConfig, RapidOCRPDFLoader

pdf = Path("data/医学影像学学习指导与习题集.pdf")
ocr_config = OCRConfig(cache_dir=Path("artifacts/ocr"), scale=1.0, min_confidence=0.5, min_chars=50, use_cls=False)
pages = RapidOCRPDFLoader(pdf, ocr_config).load()

# Case-analysis answers live near the end (附录病例分析 答案). Scan pages
# that contain "病例分析" AND "参考答案" OR look for case-id 【病例2-1】 near
# answer-like text. Dump a range of likely answer pages.
for p in pages:
    if 195 <= p.page_number <= 205:
        if "病例" in p.text:
            print(f"\n########## page {p.page_number} ##########")
            print(p.text[:1600])