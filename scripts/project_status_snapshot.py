"""Report and verify the current project/index/document status.

The evidence index is the source of truth for corpus statistics. Test totals
come from pytest collection, while an optional full pytest run verifies the
documented passed/skipped split.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class IndexSnapshot:
    total_evidence: int
    unique_evidence_ids: int
    duplicate_id_groups: int
    duplicate_records: int
    evidence_types: dict[str, int]
    source_files: dict[str, int]
    image_kinds: dict[str, int]


@dataclass(frozen=True)
class TestStatus:
    passed: int
    skipped: int

    @property
    def total(self) -> int:
        return self.passed + self.skipped


def build_index_snapshot(evidence_path: str | Path) -> IndexSnapshot:
    path = Path(evidence_path)
    evidence_types: Counter[str] = Counter()
    source_files: Counter[str] = Counter()
    image_kinds: Counter[str] = Counter()
    evidence_ids: Counter[str] = Counter()
    total = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            evidence_id = str(item.get("evidence_id", "")).strip()
            if not evidence_id:
                raise ValueError(f"Missing evidence_id at {path}:{line_number}")
            total += 1
            evidence_ids[evidence_id] += 1
            evidence_types[str(item.get("evidence_type", ""))] += 1
            source_files[str(item.get("source_file", ""))] += 1
            if item.get("evidence_type") == "image_caption":
                image_kind = str((item.get("metadata") or {}).get("image_kind", ""))
                if image_kind:
                    image_kinds[image_kind] += 1

    duplicates = [count for count in evidence_ids.values() if count > 1]
    return IndexSnapshot(
        total_evidence=total,
        unique_evidence_ids=len(evidence_ids),
        duplicate_id_groups=len(duplicates),
        duplicate_records=sum(count - 1 for count in duplicates),
        evidence_types=dict(sorted(evidence_types.items())),
        source_files=dict(sorted(source_files.items())),
        image_kinds=dict(sorted(image_kinds.items())),
    )


def collect_test_count(repo_root: str | Path) -> int:
    result = _run_pytest(repo_root, "--collect-only", "-q")
    match = re.search(r"(\d+) tests? collected", result)
    if not match:
        raise RuntimeError("Could not parse pytest collection count.")
    return int(match.group(1))


def run_test_suite(repo_root: str | Path) -> TestStatus:
    output = _run_pytest(repo_root, "-q")
    match = re.search(
        r"(?P<passed>\d+) passed(?:, (?P<skipped>\d+) skipped)?",
        output,
    )
    if not match:
        raise RuntimeError("Could not parse pytest pass/skip summary.")
    return TestStatus(
        passed=int(match.group("passed")),
        skipped=int(match.group("skipped") or 0),
    )


def _run_pytest(repo_root: str | Path, *arguments: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *arguments],
        cwd=Path(repo_root),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pytest {' '.join(arguments)} failed:\n{result.stdout}")
    return result.stdout


def read_pdf_text(path: str | Path) -> str:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - project requirements include PyMuPDF.
        raise RuntimeError("PyMuPDF is required to verify the resume PDF.") from exc

    with fitz.open(Path(path)) as document:
        return "\n".join(page.get_text() for page in document)


def parse_documented_test_status(text: str, document_kind: str) -> TestStatus:
    patterns = {
        "readme": r"Test suite status in the project environment:\s*(\d+) passed,\s*(\d+) skipped",
        "html": r"当前测试：\s*(\d+) passed\s*/\s*(\d+) skipped",
        "pdf": r"(\d+)\s*项通过[，,、]\s*(\d+)\s*项\s*跳过",
    }
    match = re.search(patterns[document_kind], text, flags=re.IGNORECASE)
    if not match:
        raise AssertionError(f"Missing test status in {document_kind}.")
    return TestStatus(passed=int(match.group(1)), skipped=int(match.group(2)))


def verify_materials(
    snapshot: IndexSnapshot,
    collected_tests: int,
    *,
    readme_path: str | Path,
    html_path: str | Path,
    resume_pdf_path: str | Path,
    runtime_status: TestStatus | None = None,
) -> dict[str, TestStatus]:
    readme = Path(readme_path).read_text(encoding="utf-8")
    html = Path(html_path).read_text(encoding="utf-8")
    pdf_text = read_pdf_text(resume_pdf_path)
    compact_pdf = re.sub(r"\s+", " ", pdf_text)

    _require_fragments(
        readme,
        (
            f"{snapshot.total_evidence:,}",
            "text",
            f"{snapshot.evidence_types['text']:,}",
            "exercise_qa",
            f"{snapshot.evidence_types['exercise_qa']:,}",
            "image_caption",
            f"{snapshot.evidence_types['image_caption']:,}",
            "0 duplicate evidence-id groups",
        ),
        "README",
    )
    _require_fragments(
        html,
        (
            f"{snapshot.total_evidence:,}",
            f"{snapshot.evidence_types['text']:,} 条正文",
            f"{snapshot.evidence_types['exercise_qa']:,} 条习题",
            f"{snapshot.evidence_types['image_caption']:,} 条图片",
            "0 组重复 ID",
        ),
        "HTML handbook",
    )
    _require_fragments(
        compact_pdf,
        (
            f"{snapshot.evidence_types['text']:,} 条正文",
            f"{snapshot.evidence_types['exercise_qa']:,} 条习题 QA",
            f"{snapshot.evidence_types['image_caption']:,} 条",
            f"{snapshot.total_evidence:,} 条可追溯证据",
        ),
        "resume PDF",
    )

    stale_current_claims = (
        "still contains 21 duplicate",
        "saved artifact still has 21 duplicate",
        "旧版 README仍写着",
        "当前 artifact 有 21 组重复",
    )
    combined = "\n".join((readme, html, pdf_text))
    found_stale = [claim for claim in stale_current_claims if claim in combined]
    if found_stale:
        raise AssertionError(f"Stale current-state claims remain: {found_stale}")
    if snapshot.duplicate_id_groups:
        raise AssertionError(
            f"Evidence index has {snapshot.duplicate_id_groups} duplicate ID groups."
        )

    statuses = {
        "readme": parse_documented_test_status(readme, "readme"),
        "html": parse_documented_test_status(html, "html"),
        "pdf": parse_documented_test_status(pdf_text, "pdf"),
    }
    if len(set(statuses.values())) != 1:
        raise AssertionError(f"Documented test statuses disagree: {statuses}")
    documented = next(iter(statuses.values()))
    if documented.total != collected_tests:
        raise AssertionError(
            f"Documents report {documented.total} tests, but pytest collects {collected_tests}."
        )
    if runtime_status is not None and documented != runtime_status:
        raise AssertionError(
            f"Documents report {documented}, but pytest completed with {runtime_status}."
        )
    return statuses


def _require_fragments(text: str, fragments: Iterable[str], label: str) -> None:
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise AssertionError(f"{label} is missing current status fragments: {missing}")


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    desktop = Path.home() / "Desktop"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=repo_root / "artifacts" / "index" / "evidence.jsonl",
    )
    parser.add_argument("--readme", type=Path, default=repo_root / "README.md")
    parser.add_argument(
        "--html",
        type=Path,
        default=desktop / "project-status-report.html",
    )
    parser.add_argument(
        "--resume-pdf",
        type=Path,
        default=desktop / "project-notes.pdf",
    )
    parser.add_argument(
        "--check-docs",
        action="store_true",
        help="Verify README, HTML handbook, and resume PDF against the snapshot.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run pytest and verify the documented passed/skipped split.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    snapshot = build_index_snapshot(args.evidence)
    collected_tests = collect_test_count(repo_root)
    runtime_status = run_test_suite(repo_root) if args.run_tests else None
    statuses = None
    if args.check_docs:
        statuses = verify_materials(
            snapshot,
            collected_tests,
            readme_path=args.readme,
            html_path=args.html,
            resume_pdf_path=args.resume_pdf,
            runtime_status=runtime_status,
        )

    payload = {
        "index": asdict(snapshot),
        "collected_tests": collected_tests,
        "runtime_test_status": asdict(runtime_status) if runtime_status else None,
        "document_test_status": (
            {name: asdict(status) for name, status in statuses.items()}
            if statuses
            else None
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
