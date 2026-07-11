"""Generate a reproducible Markdown report for exercise PDF parsing."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import _is_exercise_book, _load_pages
from src.config.settings import load_settings
from src.document.exercise_parser import ExerciseParser
from src.schema import ExerciseParseReport, ExerciseQAChunk


DEFAULT_OUTPUT = Path("experiments/results/exercise_parse_report.md")


@dataclass(frozen=True)
class ExerciseBookStats:
    """Aggregated parser statistics for one exercise PDF."""

    source_file: str
    chunks: int
    unmatched_questions: int
    unmatched_answers: int
    estimated_pairing_rate: float
    by_chapter: tuple[tuple[str, int], ...]
    by_question_type: tuple[tuple[str, int], ...]
    by_chapter_and_type: tuple[tuple[str, str, int], ...]
    unmatched_samples: tuple[tuple[str, str], ...]


def build_book_stats(pdf_path: Path, report: ExerciseParseReport) -> ExerciseBookStats:
    """Build stable, serializable statistics from an ExerciseParseReport."""

    chapter_counts = Counter(_display_value(chunk.chapter) for chunk in report.chunks)
    type_counts = Counter(_display_value(chunk.question_type) for chunk in report.chunks)
    chapter_type_counts = Counter(
        (_display_value(chunk.chapter), _display_value(chunk.question_type)) for chunk in report.chunks
    )

    matched_chunks = len(report.chunks) - len(report.unmatched_questions)
    pairing_denominator = matched_chunks + len(report.unmatched_questions) + len(report.unmatched_answers)
    pairing_rate = matched_chunks / pairing_denominator if pairing_denominator else 0.0

    samples = _unmatched_samples(report)
    return ExerciseBookStats(
        source_file=pdf_path.name,
        chunks=len(report.chunks),
        unmatched_questions=len(report.unmatched_questions),
        unmatched_answers=len(report.unmatched_answers),
        estimated_pairing_rate=pairing_rate,
        by_chapter=tuple(chapter_counts.most_common()),
        by_question_type=tuple(type_counts.most_common()),
        by_chapter_and_type=tuple(
            (chapter, question_type, count)
            for (chapter, question_type), count in sorted(
                chapter_type_counts.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
        ),
        unmatched_samples=samples,
    )


def render_markdown_report(stats: list[ExerciseBookStats]) -> str:
    """Render parser statistics as a Markdown report."""

    lines = [
        "# Exercise PDF Parse Report",
        "",
        "This report is generated from local PDFs with `ExerciseParser`.",
        "",
        "Pairing rate estimate: `matched_chunks / (matched_chunks + unmatched_questions + unmatched_answers)`.",
        "",
    ]
    if not stats:
        lines.extend(["No exercise PDFs were found.", ""])
        return "\n".join(lines)

    total_chunks = sum(item.chunks for item in stats)
    total_unmatched_questions = sum(item.unmatched_questions for item in stats)
    total_unmatched_answers = sum(item.unmatched_answers for item in stats)
    total_matched = total_chunks - total_unmatched_questions
    denominator = total_matched + total_unmatched_questions + total_unmatched_answers
    total_pairing_rate = total_matched / denominator if denominator else 0.0

    lines.extend(
        [
            "## Summary",
            "",
            "| PDF | Chunks | Unmatched Questions | Unmatched Answers | Estimated Pairing Rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in stats:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(item.source_file),
                    str(item.chunks),
                    str(item.unmatched_questions),
                    str(item.unmatched_answers),
                    _percent(item.estimated_pairing_rate),
                ]
            )
            + " |"
        )
    lines.append(
        "| "
        + " | ".join(
            [
                "**Total**",
                str(total_chunks),
                str(total_unmatched_questions),
                str(total_unmatched_answers),
                _percent(total_pairing_rate),
            ]
        )
        + " |"
    )
    lines.append("")

    for item in stats:
        lines.extend(
            [
                f"## {_md(item.source_file)}",
                "",
                "### Chapter Distribution",
                "",
                "| Chapter | Chunks |",
                "|---|---:|",
            ]
        )
        lines.extend(_count_rows(item.by_chapter))
        lines.extend(
            [
                "",
                "### Question Type Distribution",
                "",
                "| Question Type | Chunks |",
                "|---|---:|",
            ]
        )
        lines.extend(_count_rows(item.by_question_type))
        lines.extend(
            [
                "",
                "### Chapter And Question Type Distribution",
                "",
                "| Chapter | Question Type | Chunks |",
                "|---|---|---:|",
            ]
        )
        if item.by_chapter_and_type:
            for chapter, question_type, count in item.by_chapter_and_type:
                lines.append(f"| {_md(chapter)} | {_md(question_type)} | {count} |")
        else:
            lines.append("| _(none)_ | _(none)_ | 0 |")
        lines.extend(
            [
                "",
                "### Unmatched Samples",
                "",
                "| Kind | Sample |",
                "|---|---|",
            ]
        )
        if item.unmatched_samples:
            for kind, sample in item.unmatched_samples:
                lines.append(f"| {_md(kind)} | {_md(sample)} |")
        else:
            lines.append("| _(none)_ | _(none)_ |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def generate_report(config_path: Path, output_path: Path, *, ocr_mode: str = "auto") -> list[ExerciseBookStats]:
    """Parse local exercise PDFs and write the Markdown report."""

    settings = load_settings(config_path)
    data_dir = Path(settings.get("paths", "data_dir", default="data"))
    pdf_paths = sorted(path for path in data_dir.glob("*.pdf") if _is_exercise_book(path))

    stats: list[ExerciseBookStats] = []
    parser = ExerciseParser()
    for pdf_path in pdf_paths:
        pages = _load_pages(pdf_path, settings, ocr_mode=ocr_mode)
        stats.append(build_book_stats(pdf_path, parser.parse(pages)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown_report(stats), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report exercise PDF parsing statistics.")
    parser.add_argument("--config", default="config.yaml", help="Path to local YAML config.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Markdown report output path.")
    parser.add_argument("--ocr", choices=["auto", "never"], default="auto", help="Whether to apply OCR fallback.")
    args = parser.parse_args(argv)

    stats = generate_report(Path(args.config), Path(args.output), ocr_mode=args.ocr)
    print(f"Wrote {args.output} for {len(stats)} exercise PDF(s).")
    return 0


def _unmatched_samples(report: ExerciseParseReport) -> tuple[tuple[str, str], ...]:
    samples: list[tuple[str, str]] = []
    for chunk in report.unmatched_questions:
        samples.append(("question", _question_sample(chunk)))
    for answer in report.unmatched_answers:
        samples.append(("answer", _truncate(answer)))
    return tuple(samples[:10])


def _question_sample(chunk: ExerciseQAChunk) -> str:
    parts = [
        chunk.question_id,
        f"page={chunk.page_question}" if chunk.page_question is not None else "",
        chunk.question_type,
        chunk.question_text,
    ]
    return _truncate(" | ".join(part for part in parts if part))


def _count_rows(items: tuple[tuple[str, int], ...]) -> list[str]:
    if not items:
        return ["| _(none)_ | 0 |"]
    return [f"| {_md(label)} | {count} |" for label, count in items]


def _display_value(value: str) -> str:
    return value.strip() if value and value.strip() else "未标注"


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _truncate(value: str, limit: int = 180) -> str:
    normalized = " ".join(str(value).split())
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "..."


def _md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
