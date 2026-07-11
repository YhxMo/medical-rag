"""Structured parsing for exercise books with separated answers."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.schema import ExerciseParseReport, ExerciseQAChunk, PageDocument


_CHAPTER_RE = re.compile(r"^\s*(第[一二三四五六七八九十百千万\d]+[章节篇][^\n]*)")
_QUESTION_RE = re.compile(r"^\s*(\d+)[\.．、)]\s*(.*)")
_OPTION_RE = re.compile(r"^\s*[A-H][\.．、:：]\s*.+")
_QUESTION_TYPE_RE = re.compile(
    r"(名词解释|填空题|选择题|简答题|病例分析|A1型题|A2型题|A3/A4型题|B型题|问答题|论述题)"
)
_COMPACT_ANSWER_RE = re.compile(
    r"(?<!\d)(\d{1,3})\s*[\.\、．]?\s*([A-H](?:[A-H])?)(?=\s*\d{1,3}\s*[\.\、．]?\s*[A-H]|$)"
)
# Case-analysis identifier like 【病例2-1】 — chapter-relative case number.
# Case-analysis questions are laid out as 【病例X-Y】 then a stem then 【提问】
# then 1./2./3. sub-questions; answers reuse the same 【病例X-Y】 id. Matching
# rides on (chapter, section, 病例分析, "X-Y:sub") instead of a bare number,
# because every case restarts sub-question numbering from 1.
_CASE_RE = re.compile(r"【\s*病例\s*(\d+)\s*[-–—]\s*(\d+)\s*】")


def _is_chapter_or_part_heading(line: str) -> bool:
    """True for 第X章 / 第X篇 headings, False for 第X节 (section).

    Answer keys use 第X节 sub-headings (e.g. "第一节眼部") *inside* a chapter's
    answer block; those must not be mistaken for chapter starts.
    """
    match = re.search(r"第[一二三四五六七八九十百千万\d]+([章节篇])", line)
    if not match:
        return False
    return match.group(1) in ("章", "篇")


def _strip_repeated_headers(pages: list[PageDocument]) -> list[list[str]]:
    """Return per-page line lists with running-header chapter titles removed.

    These exercise books repeat the chapter title as the first line of every
    page (a page header). The first time a chapter heading appears it is real
    book content (the chapter title itself); on each subsequent page of that
    chapter the same heading re-appears as a running header and is pure noise
    — it would make the parser think every page starts a new chapter and chop
    questions/answers apart. We drop those repeats here so the parser sees one
    clean chapter heading at the true start of each chapter.

    Two headings are treated as "the same chapter's header" when they share
    the same 第X章 / 第X篇 token, even if the trailing title text differs
    (e.g. "第三章" on one page and "第三章头颈部" on the next are the same
    chapter). Lines that aren't chapter/part headings (第X章/第X篇) are never
    stripped, even when they sit on the first line of a page.
    """
    cleaned: list[list[str]] = []
    last_chapter_key: str | None = None
    for page in pages:
        lines = page.text.splitlines()
        first_nonempty_idx = None
        for idx, raw in enumerate(lines):
            if raw.strip():
                first_nonempty_idx = idx
                break
        if first_nonempty_idx is not None:
            first = lines[first_nonempty_idx].strip()
            if _CHAPTER_RE.match(first) and _is_chapter_or_part_heading(first):
                key = _chapter_key(first)
                if key == last_chapter_key:
                    # Running header for the chapter we're already inside → drop it.
                    del lines[first_nonempty_idx]
                else:
                    # A genuinely new chapter heading — keep it.
                    last_chapter_key = key
        cleaned.append(lines)
    return cleaned


@dataclass
class _QuestionDraft:
    chapter: str
    number: str
    text_lines: list[str]
    options: list[str]
    page: int
    question_type: str = ""
    section: str = ""


@dataclass
class _AnswerDraft:
    chapter: str
    number: str
    answer_lines: list[str]
    page: int
    question_type: str = ""
    section: str = ""


class ExerciseParser:
    """Parse exercise PDFs into one-question-one-chunk records."""

    def __init__(
        self,
        *,
        question_markers: tuple[str, ...] = ("习题", "选择题", "名词解释", "简答题", "病例分析"),
        answer_markers: tuple[str, ...] = ("参考答案", "答案", "解析", "参考答案与解析"),
    ) -> None:
        self.question_markers = question_markers
        self.answer_markers = answer_markers

    def parse(self, pages: list[PageDocument]) -> ExerciseParseReport:
        questions: list[_QuestionDraft] = []
        answers: list[_AnswerDraft] = []
        current_question: _QuestionDraft | None = None
        current_answer: _AnswerDraft | None = None
        current_chapter = "未分章"
        current_section_title = ""
        current_type = ""
        current_case = ""
        case_stem_lines: list[str] = []
        section = "question"

        def finish_question() -> None:
            nonlocal current_question
            if current_question is not None:
                questions.append(current_question)
                current_question = None

        def finish_answer() -> None:
            nonlocal current_answer
            if current_answer is not None:
                answers.append(current_answer)
                current_answer = None

        for page, lines in zip(pages, _strip_repeated_headers(pages)):
            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue

                chapter_match = _CHAPTER_RE.match(line)
                if chapter_match:
                    is_chapter = _is_chapter_or_part_heading(line)
                    heading_key = _chapter_key(chapter_match.group(1))
                    is_new_chapter = is_chapter and heading_key != _chapter_key(current_chapter)
                    heading_text = chapter_match.group(1)
                    if is_chapter:
                        current_chapter = heading_text
                        # A chapter heading resets the section (第X节) context.
                        current_section_title = ""
                    else:
                        # Sub-section (第X节) title — remember its text so we
                        # can disambiguate identically-numbered questions that
                        # repeat within a chapter across its sections (e.g.
                        # 第三章 has 7 sub-sections, each starting 填空题 from
                        # num=1). Does not update current_chapter.
                        current_section_title = heading_text
                    # Both a *new* chapter heading (第X章/第X篇) and any
                    # section heading (第X节) can mark the start of a fresh
                    # round of questions in this book: each "第X节" block
                    # within a chapter follows the same 复习思考题 → 参考答案
                    # layout as a chapter. So when we hit either while inside
                    # an answer block, switch back to the question section.
                    #
                    # Caveats:
                    # - Section (第X节) headings do NOT update current_chapter,
                    #   so (chapter_key, type, number) pairing still keys
                    #   questions and answers by the enclosing chapter.
                    # - Only a *new* chapter (different 第X章 token) really
                    #   flips the chapter. A re-printed same-chapter title
                    #   inside an answer block must NOT flip us, or the answer
                    #   text would be mis-parsed as a question.
                    flip = (is_chapter and is_new_chapter) or not is_chapter
                    if section == "answer" and flip:
                        finish_question()
                        finish_answer()
                        section = "question"
                        current_type = ""
                        current_case = ""
                    continue

                case_match = _CASE_RE.search(line)
                if case_match:
                    # 【病例X-Y】 marks the start of a case-analysis question
                    # (in the question section) or its answer (in the answer
                    # section). Either way it starts a new record, so flush
                    # the in-progress one and latch the case id; the type is
                    # 病例分析 and subsequent 1./2./3. lines attach to it.
                    finish_question()
                    finish_answer()
                    current_case = f"{case_match.group(1)}-{case_match.group(2)}"
                    current_type = "病例分析"
                    case_stem_lines = []
                    continue

                if self._is_answer_marker(line):
                    finish_question()
                    finish_answer()
                    section = "answer"
                    current_type = ""
                    current_case = ""
                    continue

                if self._is_question_section_marker(line):
                    finish_question()
                    finish_answer()
                    section = "question"
                    current_type = ""
                    current_case = ""
                    continue

                question_type = self._question_type(line)
                if question_type:
                    finish_question()
                    finish_answer()
                    current_type = question_type
                    # A new question-type marker ends any in-progress case.
                    current_case = ""
                    if section != "answer":
                        section = "question"
                    continue

                question_match = _QUESTION_RE.match(line)
                if question_match and section == "question":
                    # Only collect a question when a question-type marker
                    # (名词解释 / 选择题 / A1型题 / …) has set current_type.
                    # Exercise books open each chapter with a "一、重点和难点"
                    # block whose "1. 要点 / 2. 要点" numbered list looks exactly
                    # like a question stem to _QUESTION_RE but is study-point
                    # prose, not a real question — collecting it produces
                    # unmatchable noise questions and pollutes the index.
                    if not current_type:
                        continue
                    finish_question()
                    # Case-analysis sub-questions repeat 1./2./3. per case, so
                    # namespace the number under the current case id to keep
                    # (chapter, section, type, number) unique and matchable.
                    number = f"{current_case}:{question_match.group(1)}" if current_case else question_match.group(1)
                    # Prepend the case stem (patient description etc.) captured
                    # between 【病例X-Y】 and the first sub-question, so each
                    # sub-question chunk carries the clinical context it asks
                    # about. Reset the buffer after the first sub-question.
                    stem = case_stem_lines if current_case else []
                    case_stem_lines = []
                    current_question = _QuestionDraft(
                        chapter=current_chapter,
                        number=number,
                        text_lines=stem + [question_match.group(2).strip()],
                        options=[],
                        page=page.page_number,
                        question_type=current_type,
                        section=current_section_title,
                    )
                    continue

                if section == "answer":
                    compact_answers = _parse_compact_answers(line)
                    if compact_answers:
                        finish_answer()
                        for number, answer_text in compact_answers:
                            namespaced = f"{current_case}:{number}" if current_case else number
                            answers.append(
                                _AnswerDraft(
                                    chapter=current_chapter,
                                    number=namespaced,
                                    answer_lines=[answer_text],
                                    page=page.page_number,
                                    question_type=current_type,
                                    section=current_section_title,
                                )
                            )
                        continue

                    if question_match:
                        finish_answer()
                        number = f"{current_case}:{question_match.group(1)}" if current_case else question_match.group(1)
                        current_answer = _AnswerDraft(
                            chapter=current_chapter,
                            number=number,
                            answer_lines=[question_match.group(2).strip()],
                            page=page.page_number,
                            question_type=current_type,
                            section=current_section_title,
                        )
                        continue

                if section == "question" and current_question is not None:
                    if _OPTION_RE.match(line):
                        current_question.options.append(line)
                    else:
                        current_question.text_lines.append(line)
                    continue

                # Between 【病例X-Y】 and the first 1./2./3. sub-question, the
                # patient/stem lines and 【提问】 marker have no current_question
                # to attach to yet — buffer them as the case stem. Skip the
                # literal "【提问】" marker; it carries no information.
                if section == "question" and current_case and not current_question:
                    if "【提问】" not in line:
                        case_stem_lines.append(line)
                    continue

                if section == "answer" and current_answer is not None:
                    current_answer.answer_lines.append(line)

        finish_question()
        finish_answer()

        source_file = pages[0].source_file if pages else "unknown.pdf"
        source_stem = Path(source_file).stem
        answer_map = {
            (_chapter_key(answer.chapter), _section_key(answer.section), answer.question_type, answer.number): answer
            for answer in answers
        }
        fallback_answer_map: dict[tuple[str, str, str], list[_AnswerDraft]] = defaultdict(list)
        for answer in answers:
            fallback_answer_map[(_chapter_key(answer.chapter), _section_key(answer.section), answer.number)].append(answer)
        duplicate_question_numbers = Counter(
            (_chapter_key(question.chapter), _section_key(question.section), question.number)
            for question in questions
        )
        base_question_ids = [
            self._question_id(
                source_stem,
                question,
                duplicate_question_numbers[
                    (_chapter_key(question.chapter), _section_key(question.section), question.number)
                ]
                > 1,
            )
            for question in questions
        ]
        duplicate_question_ids = Counter(base_question_ids)
        question_id_occurrences: Counter[str] = Counter()
        chunks: list[ExerciseQAChunk] = []
        unmatched_questions: list[ExerciseQAChunk] = []

        for question, base_question_id in zip(questions, base_question_ids):
            chapter_key = _chapter_key(question.chapter)
            section_key = _section_key(question.section)
            answer = answer_map.pop((chapter_key, section_key, question.question_type, question.number), None)
            if answer is None:
                # Last-ditch fallback: same chapter+section+number, ignoring
                # question_type. Answers occasionally drop their type marker
                # via OCR; this still pairs them as long as the (section,
                # number) is unique within the chapter.
                candidates = fallback_answer_map.get((chapter_key, section_key, question.number), [])
                if len(candidates) == 1:
                    answer = candidates[0]
                    answer_map.pop(
                        (_chapter_key(answer.chapter), _section_key(answer.section), answer.question_type, answer.number),
                        None,
                    )
            include_type_in_id = duplicate_question_numbers[(chapter_key, section_key, question.number)] > 1
            occurrence_index: int | None = None
            if duplicate_question_ids[base_question_id] > 1:
                question_id_occurrences[base_question_id] += 1
                occurrence_index = question_id_occurrences[base_question_id]
            chunk = self._build_chunk(
                source_file,
                source_stem,
                question,
                answer,
                include_type_in_id,
                occurrence_index,
            )
            if answer is None:
                unmatched_questions.append(chunk)
            chunks.append(chunk)

        unmatched_answers = tuple(
            f"{answer.chapter}:{answer.number}:{' '.join(answer.answer_lines).strip()}"
            for answer in answer_map.values()
        )
        return ExerciseParseReport(
            chunks=tuple(chunks),
            unmatched_questions=tuple(unmatched_questions),
            unmatched_answers=unmatched_answers,
        )

    @staticmethod
    def _is_section_marker(line: str, markers: tuple[str, ...]) -> bool:
        normalized = line.strip()
        compound_markers = {"参考答案与解析", "答案与解析"}
        return normalized in set(markers) or normalized in compound_markers

    def _is_answer_marker(self, line: str) -> bool:
        normalized = line.strip()
        if "参考答案" in normalized or "答案与解析" in normalized:
            return True
        if normalized in {"答案", "解析"}:
            return True
        return False

    def _is_question_section_marker(self, line: str) -> bool:
        normalized = line.strip()
        if "参考答案" in normalized or "答案" in normalized:
            return False
        return bool(re.match(r"^([一二三四五六七八九十]+[、.．])?\s*习题$", normalized))

    def _question_type(self, line: str) -> str:
        normalized = line.strip()
        if not normalized or len(normalized) > 30:
            return ""
        match = _QUESTION_TYPE_RE.search(normalized)
        if not match:
            return ""
        return match.group(1)

    @staticmethod
    def _build_chunk(
        source_file: str,
        source_stem: str,
        question: _QuestionDraft,
        answer: _AnswerDraft | None,
        include_type_in_id: bool = False,
        occurrence_index: int | None = None,
    ) -> ExerciseQAChunk:
        question_text = "\n".join(item for item in question.text_lines if item).strip()
        answer_text = ""
        explanation = ""
        page_answer: int | None = None
        if answer is not None:
            joined = "\n".join(item for item in answer.answer_lines if item).strip()
            answer_text, explanation = _split_answer_explanation(joined)
            page_answer = answer.page

        question_id = ExerciseParser._question_id(source_stem, question, include_type_in_id, occurrence_index)
        metadata: dict = {}
        if question.section:
            metadata["section"] = question.section
        if occurrence_index is not None:
            metadata["question_id_occurrence"] = occurrence_index
        return ExerciseQAChunk(
            question_id=question_id,
            source_file=source_file,
            chapter=question.chapter,
            question_number=question.number,
            question_type=question.question_type,
            question_text=question_text,
            options=tuple(question.options),
            answer=answer_text,
            explanation=explanation,
            page_question=question.page,
            page_answer=page_answer,
            metadata=metadata,
        )

    @staticmethod
    def _question_id(
        source_stem: str,
        question: _QuestionDraft,
        include_type_in_id: bool = False,
        occurrence_index: int | None = None,
    ) -> str:
        id_parts = [source_stem, question.chapter]
        section_key = _section_key(question.section)
        if section_key:
            # Sections repeat question numbers within a chapter (each 第X节
            # starts 名词解释 / 填空题 / 选择题 from num=1), so the section
            # token has to be part of the id to keep it unique.
            id_parts.append(section_key)
        if include_type_in_id and question.question_type:
            id_parts.append(question.question_type)
        id_parts.append(question.number)
        if occurrence_index is not None:
            id_parts.append(f"occurrence{occurrence_index}")
        return ":".join(id_parts)


def _split_answer_explanation(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    for marker in ("解析", "说明", "理由"):
        if marker in text:
            before, after = text.split(marker, 1)
            return before.strip().strip(" ：:;；"), f"{marker}{after}".strip()
    first_line, _, rest = text.partition("\n")
    return first_line.strip(), rest.strip()


def _parse_compact_answers(line: str) -> list[tuple[str, str]]:
    normalized = line.replace(" ", "")
    matches = list(_COMPACT_ANSWER_RE.finditer(normalized))
    if len(matches) < 2:
        return []
    consumed = "".join(match.group(0) for match in matches)
    if len(consumed) < len(normalized) * 0.7:
        return []
    return [(match.group(1), match.group(2)) for match in matches]


def _chapter_key(chapter: str) -> str:
    match = re.search(r"第[一二三四五六七八九十百千万\d]+[章节篇]", chapter)
    return match.group(0) if match else chapter


def _section_key(section: str) -> str:
    """Extract the 第X节 token (e.g. "第一节"), or "" if none.

    Sections repeat question numbers within a chapter, so the (chapter, section,
    type, number) quadruple is the true unique key. Sections sit inside a
    chapter, so _chapter_key already collapses them onto the chapter; this
    function gives the orthogonal sub-chapter axis. Returns "" for chapters
    that have no 第X节 sub-divisions — keeping the four-tuple uniform.
    """
    match = re.search(r"第[一二三四五六七八九十百千万\d]+节", section)
    return match.group(0) if match else ""
