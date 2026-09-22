"""Diagnostic for unmatched questions across both exercise books."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
from collections import Counter
from src.document.ocr import OCRConfig, RapidOCRPDFLoader
from src.document.exercise_parser import ExerciseParser

def run(pdf_name):
    pdf = Path(f"data/{pdf_name}")
    ocr_config = OCRConfig(cache_dir=Path("artifacts/ocr"), scale=1.0, min_confidence=0.5, min_chars=50, use_cls=False)
    pages = RapidOCRPDFLoader(pdf, ocr_config).load()
    report = ExerciseParser().parse(pages)
    print(f"\n##### {pdf_name} #####")
    print(f"chunks={len(report.chunks)} unmatched_q={len(report.unmatched_questions)} unmatched_a={len(report.unmatched_answers)}")
    # type dist of unmatched questions
    c = Counter(chunk.question_type for chunk in report.unmatched_questions)
    print("unmatched_q by type:", dict(c))
    c2 = Counter(chunk.chapter for chunk in report.unmatched_questions)
    print("unmatched_q by chapter (top):", c2.most_common(8))
    # show some unmatched question numbers grouped
    nums = Counter(chunk.question_number for chunk in report.unmatched_questions)
    odd = [(n, v) for n, v in nums.most_common() if not re.fullmatch(r"\d{1,3}", n) or v > 3]
    print("odd/duplicate question numbers in unmatched (top 15):", odd[:15])
    print("sample unmatched questions (first 8):")
    for chunk in report.unmatched_questions[:8]:
        print(f"  [chap={chunk.chapter!r} type={chunk.question_type!r} num={chunk.question_number!r} p{chunk.page_question}] {chunk.question_text[:70]}")

run("医学影像学学习指导与习题集.pdf")
run("医学影像诊断学习题集.pdf")