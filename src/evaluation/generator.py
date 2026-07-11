"""Semi-automatic draft evaluation question generation."""
from __future__ import annotations

import re

from src.evaluation.dataset import EvaluationQuestion
from src.schema import EvidenceItem


_CJK_PATTERN = re.compile(r"[一-鿿]")
_ALNUM_PATTERN = re.compile(r"[A-Za-z0-9]")
_MIN_MEANINGFUL_CHARS = 40
_MIN_MEANINGFUL_RATIO = 0.5

_REFERENCE_LIST_MARKER = "推荐阅读"
_CITATION_ENTRY_PATTERN = re.compile(r"［\d+］|\[\d+\]")
_NAVIGATION_NOISE_LINES = ("本章数字资源", "本章思维导图")


def generate_draft_questions(evidence: list[EvidenceItem], *, limit: int = 30) -> list[EvaluationQuestion]:
    usable = [item for item in evidence if _is_usable(item)]
    sampled = _sample_evenly(usable, limit)

    questions: list[EvaluationQuestion] = []
    for item in sampled:
        if item.evidence_type == "exercise_qa":
            question_text = item.metadata.get("question_text") or _extract_question(item.content)
            question = question_text if question_text else f"请回答这道习题：{item.evidence_id}"
        else:
            snippet = _strip_navigation_noise(item.content)[:60]
            question = f"请概括这段内容的医学影像学要点：{snippet}"
        questions.append(
            EvaluationQuestion(
                question_id=f"eval_{len(questions) + 1:04d}",
                question=question,
                expected_evidence_ids=(item.evidence_id,),
                reviewed=False,
                metadata={"source_file": item.source_file, "evidence_type": item.evidence_type},
            )
        )
    return questions


def _is_usable(item: EvidenceItem) -> bool:
    """Filter out low-signal evidence such as OCR-mangled directory/watermark pages,
    end-of-chapter reference lists, and pure navigation elements.

    Exercise chunks are already structured (question/answer/explanation fields),
    so they are kept as-is regardless of raw content shape.
    """
    if item.evidence_type == "exercise_qa":
        return True
    content = item.content.strip()
    if not content:
        return False
    if _REFERENCE_LIST_MARKER in content[:20]:
        return False
    if len(_CITATION_ENTRY_PATTERN.findall(content)) >= 2:
        return False
    meaningful = len(_CJK_PATTERN.findall(content)) + len(_ALNUM_PATTERN.findall(content))
    if meaningful < _MIN_MEANINGFUL_CHARS:
        return False
    return meaningful / len(content) >= _MIN_MEANINGFUL_RATIO


def _strip_navigation_noise(content: str) -> str:
    """Drop leading chapter-navigation lines (page numbers, digital resources/mind
    map markers) so the generated question snippet leads with actual body text."""
    lines = content.splitlines()
    start = 0
    while start < len(lines):
        stripped = lines[start].strip()
        if stripped in _NAVIGATION_NOISE_LINES or stripped.isdigit():
            start += 1
            continue
        break
    return "\n".join(lines[start:]) if start else content


def _sample_evenly(evidence: list[EvidenceItem], limit: int) -> list[EvidenceItem]:
    """Spread the sample across the whole evidence list instead of only the head."""
    if limit <= 0 or not evidence:
        return []
    if len(evidence) <= limit:
        return evidence
    step = len(evidence) / limit
    indices = [int(index * step) for index in range(limit)]
    return [evidence[index] for index in indices]


def _extract_question(content: str) -> str:
    for line in content.splitlines():
        if line.startswith("题目："):
            return line.removeprefix("题目：").strip()
    return ""
