"""Evaluation dataset schema and IO."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvaluationQuestion:
    """One reviewed or draft evaluation question."""

    question_id: str
    question: str
    expected_evidence_ids: tuple[str, ...] = ()
    expected_answer: str = ""
    reviewed: bool = False
    metadata: dict = field(default_factory=dict)


def load_dataset(path: str | Path, *, reviewed_only: bool = True) -> list[EvaluationQuestion]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        return []
    with dataset_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    questions = [
        EvaluationQuestion(
            question_id=item["question_id"],
            question=item["question"],
            expected_evidence_ids=tuple(item.get("expected_evidence_ids", [])),
            expected_answer=item.get("expected_answer", ""),
            reviewed=bool(item.get("reviewed", False)),
            metadata=item.get("metadata", {}),
        )
        for item in raw
    ]
    return [item for item in questions if item.reviewed] if reviewed_only else questions


def save_dataset(path: str | Path, questions: list[EvaluationQuestion]) -> None:
    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as handle:
        json.dump([_to_json(item) for item in questions], handle, ensure_ascii=False, indent=2)


def _to_json(question: EvaluationQuestion) -> dict:
    data = asdict(question)
    data["expected_evidence_ids"] = list(question.expected_evidence_ids)
    return data
