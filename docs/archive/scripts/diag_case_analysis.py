"""Inspect 病例分析 question/answer blocks in the exercise book."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.document.ocr import OCRConfig, RapidOCRPDFLoader

pdf = Path("data/医学影像学学习指导与习题集.pdf")
ocr_config = OCRConfig(cache_dir=Path("artifacts/ocr"), scale=1.0, min_confidence=0.5, min_chars=50, use_cls=False)
pages = RapidOCRPDFLoader(pdf, ocr_config).load()

# Find pages containing 病例分析 and dump their context.
import re
shown = 0
for i, p in enumerate(pages):
    if "病例分析" in p.text and ("【病例" in p.text or "病例1" in p.text or "病例2" in p.text):
        print(f"\n########## page {p.page_number} ##########")
        print(p.text[:1800])
        shown += 1
        if shown >= 4:
            break

# Also dump a known-unmatched 病例分析 answer context: chapter 9 病例分析 num=4 was unmatched.
print("\n========== searching 病例分析 answer area (page ~241) ==========")
for p in pages:
    if 235 <= p.page_number <= 245 and "病例分析" in p.text:
        print(f"\n--- page {p.page_number} ---")
        print(p.text[:1200])
        break