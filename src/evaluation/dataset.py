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
        raise ValueError("Evaluation dataset does not exist")
    with dataset_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError("Evaluation dataset must be a list")
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Evaluation entries must be objects")
        qid, question = item.get("question_id"), item.get("question")
        if not isinstance(qid, str) or not qid.strip() or qid in seen:
            raise ValueError("Question IDs must be nonempty and unique")
        seen.add(qid)
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question text must be nonempty")
        if not isinstance(item.get("reviewed", False), bool):
            raise ValueError("reviewed must be a JSON boolean")
        labels = item.get("expected_evidence_ids", [])
        if not isinstance(labels, list) or any(
            not isinstance(x, str) or not x.strip() for x in labels
        ):
            raise ValueError("Evidence labels must be a list of nonempty strings")
        if len(labels) != len(set(labels)):
            raise ValueError("Evidence labels must be unique within each question")
        if item.get("reviewed") and not labels:
            raise ValueError("Reviewed questions require evidence labels")
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
    selected = [item for item in questions if item.reviewed] if reviewed_only else questions
    if reviewed_only and not selected:
        raise ValueError("Evaluation dataset contains no reviewed questions")
    return selected


def save_dataset(path: str | Path, questions: list[EvaluationQuestion]) -> None:
    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as handle:
        json.dump([_to_json(item) for item in questions], handle, ensure_ascii=False, indent=2)


def _to_json(question: EvaluationQuestion) -> dict:
    data = asdict(question)
    data["expected_evidence_ids"] = list(question.expected_evidence_ids)
    return data
