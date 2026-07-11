"""Build a source-grounded manual evaluation candidate set from evidence.jsonl."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_EVIDENCE = Path("artifacts/index/evidence.jsonl")
DEFAULT_JSON = Path("医学评估题目/manual_eval_candidates.json")
DEFAULT_MARKDOWN = Path("医学评估题目/manual_eval_candidates.md")
TARGETS = {"exercise_qa": 25, "text": 25, "image_caption": 10}

BAD_MARKERS = (
    "目录",
    "参考文献",
    "推荐阅读",
    "本章数字资源",
    "本章思维导图",
    "页码索引",
    "导航页",
    "非医学影像",
    "不包含任何具体",
    "无具体解剖",
    "无实际病变",
    "无法提供",
    "纯文本页面",
    "无附图",
    "以下是对",
    "严格依据",
    "保留疾病",
    "bilibili@",
)
QUESTION_STARTERS = ("什么", "如何", "哪", "为什么", "简述", "试述", "说明", "比较", "请", "根据")
STOP_TOPICS = {"特别", "尤其", "主要", "目前", "临床", "影像", "检查", "诊断", "其", "该", "此", "这", "则需行"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate source-grounded manual eval candidates.")
    parser.add_argument("--evidence", default=str(DEFAULT_EVIDENCE), help="Path to evidence.jsonl.")
    parser.add_argument("--json-output", default=str(DEFAULT_JSON), help="Candidate JSON output path.")
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN), help="Candidate Markdown output path.")
    args = parser.parse_args(argv)

    evidence = load_evidence(Path(args.evidence))
    candidates = build_candidates(evidence)
    json_path = Path(args.json_output)
    markdown_path = Path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(candidates), encoding="utf-8")
    print(f"Wrote {len(candidates)} candidates to {json_path} and {markdown_path}.")
    return 0


def load_evidence(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            evidence_id = item.get("evidence_id")
            if not evidence_id or evidence_id in seen:
                continue
            seen.add(evidence_id)
            items.append(item)
    return items


def build_candidates(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = {key: [] for key in TARGETS}
    builders = {
        "exercise_qa": candidate_from_exercise,
        "text": candidate_from_text,
        "image_caption": candidate_from_image_caption,
    }
    for item in evidence:
        evidence_type = item.get("evidence_type")
        if evidence_type not in by_type:
            continue
        candidate = builders[evidence_type](item)
        if candidate is not None:
            by_type[evidence_type].append(candidate)

    selected: list[dict[str, Any]] = []
    for evidence_type, target in TARGETS.items():
        pool = by_type[evidence_type]
        if len(pool) < target:
            raise RuntimeError(f"Only {len(pool)} usable {evidence_type} candidates; need {target}.")
        selected.extend(sample_evenly(pool, target))

    for index, candidate in enumerate(selected, start=1):
        candidate["question_id"] = f"manual_eval_{index:04d}"
    return selected


def candidate_from_exercise(item: dict[str, Any]) -> dict[str, Any] | None:
    metadata = item.get("metadata") or {}
    question_text = clean(metadata.get("question_text") or extract_prefixed(item.get("content", ""), "题目："))
    answer = clean(metadata.get("answer") or extract_prefixed(item.get("content", ""), "答案："))
    explanation = clean(metadata.get("explanation") or extract_prefixed(item.get("content", ""), "解析："))
    if not question_text or not answer or low_quality(question_text + answer):
        return None

    expected_answer = answer
    if len(expected_answer) <= 10 and explanation:
        expected_answer = join_answer(answer, explanation)
    elif explanation and len(expected_answer) < 80:
        expected_answer = join_answer(answer, explanation)
    expected_answer = truncate(expected_answer, 220)

    question = natural_exercise_question(question_text, metadata.get("question_type", ""))
    source_quote = quote_from_parts(
        [
            f"题目：{question_text}",
            f"答案：{answer}",
            f"解析：{explanation}" if explanation else "",
        ]
    )
    return base_candidate(
        item,
        question=question,
        expected_answer=expected_answer,
        source_quote=source_quote,
        difficulty=difficulty_for_exercise(metadata.get("question_type", ""), expected_answer),
    )


def candidate_from_text(item: dict[str, Any]) -> dict[str, Any] | None:
    content = clean(item.get("content", ""))
    if low_quality(content):
        return None
    for sentence in split_sentences(content):
        parsed = question_from_sentence(sentence)
        if parsed is None:
            continue
        question, expected_answer = parsed
        if expected_answer and expected_answer in sentence:
            return base_candidate(
                item,
                question=question,
                expected_answer=truncate(expected_answer, 220),
                source_quote=truncate(sentence, 240),
                difficulty="medium" if len(sentence) > 90 else "easy",
            )
    return None


def candidate_from_image_caption(item: dict[str, Any]) -> dict[str, Any] | None:
    metadata = item.get("metadata") or {}
    if metadata.get("image_kind") == "page" and item.get("page_start", 0) <= 10:
        return None
    content = clean(item.get("content", ""))
    if low_quality(content):
        return None
    quote = image_quote(content)
    if not quote:
        return None
    return base_candidate(
        item,
        question="根据图片描述，这张图片明确显示或描述了什么？",
        expected_answer=truncate(quote, 220),
        source_quote=truncate(quote, 240),
        difficulty="easy",
    )


def base_candidate(
    item: dict[str, Any],
    *,
    question: str,
    expected_answer: str,
    source_quote: str,
    difficulty: str,
) -> dict[str, Any]:
    return {
        "question_id": "",
        "question": question,
        "expected_answer": expected_answer,
        "expected_evidence_ids": [item["evidence_id"]],
        "evidence_type": item["evidence_type"],
        "source_file": item["source_file"],
        "page_start": item.get("page_start"),
        "page_end": item.get("page_end"),
        "difficulty": difficulty,
        "source_quote": source_quote,
        "reviewed": False,
    }


def natural_exercise_question(question_text: str, question_type: str) -> str:
    normalized = question_text.strip(" ：:。")
    if question_type == "名词解释" and len(normalized) <= 30:
        return f"什么是{normalized}？"
    if normalized.endswith(("？", "?")):
        return normalized
    if normalized.startswith(QUESTION_STARTERS) or any(token in normalized for token in ("哪", "何", "什么", "如何")):
        return normalized + ("？" if not normalized.endswith("？") else "")
    return f"根据该习题，{normalized}的答案是什么？"


def question_from_sentence(sentence: str) -> tuple[str, str] | None:
    sentence = sentence.strip(" 。；;")
    triggers = [
        ("表现为", "根据引文，{topic}表现为什么？"),
        ("可见", "根据引文，{topic}可见什么？"),
        ("用于", "根据引文，{topic}用于什么？"),
        ("包括", "根据引文，{topic}包括什么？"),
        ("是", "根据引文，{topic}是什么？"),
    ]
    for trigger, template in triggers:
        if trigger not in sentence:
            continue
        before, after = sentence.split(trigger, 1)
        topic = topic_from_before(before)
        answer = f"{trigger}{after}".strip(" ，,：:")
        if len(topic) < 2 or topic in STOP_TOPICS or topic.startswith(("这", "该", "此", "其", "前者", "后者")) or len(answer) < 8:
            continue
        return template.format(topic=topic), answer
    return None


def image_quote(content: str) -> str:
    for sentence in split_sentences(content):
        if any(marker in sentence for marker in BAD_MARKERS):
            continue
        if any(token in sentence for token in ("显示", "可见", "表现", "图像类型", "病变", "征象", "部位")):
            return sentence
    return ""


def split_sentences(content: str) -> list[str]:
    parts = re.split(r"(?<=[。；;])|\n+|---+|#+", content)
    results: list[str] = []
    for part in parts:
        sentence = clean(part).strip(" -•#")
        if 24 <= len(sentence) <= 240 and not low_quality(sentence):
            results.append(sentence)
    return results


def clean(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\b", " ").replace("�", "")
    text = re.sub(r"bilibili@[^\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def low_quality(text: str) -> bool:
    if not text or any(marker in text[:260] for marker in BAD_MARKERS):
        return True
    cjk_or_alnum = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text))
    return cjk_or_alnum < 20 or cjk_or_alnum / max(1, len(text)) < 0.55


def topic_from_before(before: str) -> str:
    cleaned = re.sub(r"^[一二三四五六七八九十\d、.．（）()\s]+", "", before.strip())
    cleaned = re.split(r"[，,。；;：:]", cleaned)[-1].strip()
    cleaned = re.sub(r"（[^）]*[A-Za-z][^）]*）", "", cleaned)
    tokens = [token for token in re.split(r"\s+", cleaned) if token]
    if tokens:
        cleaned = tokens[-1]
    cjk_runs = re.findall(r"[\u4e00-\u9fff]{2,30}", cleaned)
    if cjk_runs:
        cleaned = cjk_runs[-1]
    if len(cleaned) % 2 == 0 and cleaned[: len(cleaned) // 2] == cleaned[len(cleaned) // 2 :]:
        cleaned = cleaned[: len(cleaned) // 2]
    return truncate(cleaned, 24).strip(" 的为是")


def extract_prefixed(content: str, prefix: str) -> str:
    for line in content.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    return ""


def quote_from_parts(parts: list[str]) -> str:
    return truncate(" ".join(clean(part) for part in parts if clean(part)), 260)


def join_answer(answer: str, explanation: str) -> str:
    answer = answer.rstrip(" ，,。；;")
    explanation = explanation.lstrip(" ，,。；;")
    if not explanation:
        return answer
    if answer.endswith(("如", "包括", "为")):
        return answer + explanation
    return f"{answer}。{explanation}"


def difficulty_for_exercise(question_type: str, expected_answer: str) -> str:
    if question_type in {"名词解释", "选择题", "A1型题", "A2型题"} and len(expected_answer) <= 120:
        return "easy"
    return "medium"


def sample_evenly(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(items) <= limit:
        return list(items)
    step = len(items) / limit
    return [items[int(index * step)] for index in range(limit)]


def truncate(value: str, limit: int) -> str:
    value = clean(value)
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def render_markdown(candidates: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for item in candidates:
        counts[item["evidence_type"]] = counts.get(item["evidence_type"], 0) + 1
    lines = [
        "# Manual Evaluation Candidates",
        "",
        "All questions are source-grounded and generated only from local evidence quotes.",
        "",
        "## Summary",
        "",
        "| Evidence type | Count |",
        "|---|---:|",
    ]
    for evidence_type in TARGETS:
        lines.append(f"| {evidence_type} | {counts.get(evidence_type, 0)} |")
    lines.append("")
    for item in candidates:
        lines.extend(
            [
                f"## {item['question_id']} ({item['evidence_type']})",
                "",
                f"- reviewed: `{str(item['reviewed']).lower()}`",
                f"- difficulty: `{item['difficulty']}`",
                f"- source: `{item['source_file']}` pages {item['page_start']}-{item['page_end']}",
                f"- expected_evidence_ids: `{', '.join(item['expected_evidence_ids'])}`",
                "",
                f"**Question:** {item['question']}",
                "",
                f"**Expected answer:** {item['expected_answer']}",
                "",
                f"**Source quote:** {item['source_quote']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
