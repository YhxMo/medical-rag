"""Diagnostic: build question/answer key distributions to find unmatched-answer cause."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import Counter
from pathlib import Path
import re

from src.document.ocr import OCRConfig, RapidOCRPDFLoader

# Reuse parser internals by importing the module and monkey-wiring a capture.
import src.document.exercise_parser as ep

pdf = Path("data/医学影像学学习指导与习题集.pdf")
ocr_config = OCRConfig(
    cache_dir=Path("artifacts/ocr"), scale=1.0, min_confidence=0.5, min_chars=50, use_cls=False,
)
pages = RapidOCRPDFLoader(pdf, ocr_config).load()
out = Path("artifacts/diag_exercise_out.txt")
buf = []
def P(*a):
    buf.append(" ".join(str(x) for x in a))
P(f"total pages: {len(pages)}")
P("\n=== page 1 text ===")
P(pages[0].text[:1200] if pages else "n/a")
P("\n=== page 30 text ===")
P(pages[29].text[:800] if len(pages) > 29 else "n/a")

# Instrument: replicate ExerciseParser.parse() but capture questions/answers lists.
parser = ep.ExerciseParser()
questions: list[ep._QuestionDraft] = []
answers: list[ep._AnswerDraft] = []
current_question = None
current_answer = None
current_chapter = "未分章"
current_type = ""
section = "question"

def finish_question():
    global current_question
    if current_question is not None:
        questions.append(current_question)
        current_question = None

def finish_answer():
    global current_answer
    if current_answer is not None:
        answers.append(current_answer)
        current_answer = None

for page, lines in zip(pages, ep._strip_repeated_headers(pages)):
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        chapter_match = ep._CHAPTER_RE.match(line)
        if chapter_match:
            is_chapter = ep._is_chapter_or_part_heading(line)
            hkey = ep._chapter_key(chapter_match.group(1))
            is_new = is_chapter and hkey != ep._chapter_key(current_chapter)
            if is_chapter:
                current_chapter = chapter_match.group(1)
            flip = (is_chapter and is_new) or (not is_chapter)
            if section == "answer" and flip:
                finish_question(); finish_answer(); section = "question"; current_type = ""
            continue
        if parser._is_answer_marker(line):
            finish_question(); finish_answer(); section = "answer"; current_type = ""
            continue
        if parser._is_question_section_marker(line):
            finish_question(); finish_answer(); section = "question"; current_type = ""
            continue
        qt = parser._question_type(line)
        if qt:
            finish_question(); finish_answer(); current_type = qt
            if section != "answer": section = "question"
            continue
        qm = ep._QUESTION_RE.match(line)
        if qm and section == "question":
            finish_question()
            current_question = ep._QuestionDraft(
                chapter=current_chapter, number=qm.group(1),
                text_lines=[qm.group(2).strip()], options=[], page=page.page_number,
                question_type=current_type)
            continue
        if section == "answer":
            compact = ep._parse_compact_answers(line)
            if compact:
                finish_answer()
                for number, ans in compact:
                    answers.append(ep._AnswerDraft(chapter=current_chapter, number=number,
                        answer_lines=[ans], page=page.page_number, question_type=current_type))
                continue
            if qm:
                finish_answer()
                current_answer = ep._AnswerDraft(chapter=current_chapter, number=qm.group(1),
                    answer_lines=[qm.group(2).strip()], page=page.page_number, question_type=current_type)
                continue
        if section == "question" and current_question is not None:
            if ep._OPTION_RE.match(line):
                current_question.options.append(line)
            else:
                current_question.text_lines.append(line)
            continue
        if section == "answer" and current_answer is not None:
            current_answer.answer_lines.append(line)
finish_question(); finish_answer()

P(f"\nquestions: {len(questions)}, answers: {len(answers)}")

def ckey(c):
    m = re.search(r"第[一二三四五六七八九十百千万\d]+[章节篇]", c)
    return m.group(0) if m else c

P("\n=== question.chapter *raw* distribution (top 12) ===")
P(Counter(q.chapter for q in questions).most_common(12))
P("\n=== answer.chapter *raw* distribution (top 12) ===")
P(Counter(a.chapter for a in answers).most_common(12))
P("\n=== distinct raw chapter strings seen on questions ===")
P(sorted(set(q.chapter for q in questions))[:20])
P("\n=== distinct raw chapter strings seen on answers ===")
P(sorted(set(a.chapter for a in answers))[:30])

# Trace: list every line that matches _CHAPTER_RE, with page + which section is active
P("\n=== first 40 chapter-marker lines (page: line) ===")
cnt = 0
sec = "question"
n = 0
for page in pages:
    for raw in page.text.splitlines():
        line = raw.strip()
        if not line: continue
        if parser._is_answer_marker(line):
            sec = "answer"; n = 0; continue
        if parser._is_question_section_marker(line):
            sec = "question"; n = 0; continue
        if ep._CHAPTER_RE.match(line):
            P(f"p{page.page_number} [{sec}] {line[:80]}")
            cnt += 1
            if cnt >= 40: break
    if cnt >= 40: break

# Trace: every question-section and answer marker line
P("\n=== all question-section / answer markers ===")
for page in pages:
    for raw in page.text.splitlines():
        line = raw.strip()
        if not line: continue
        if parser._is_question_section_marker(line) or parser._is_answer_marker(line):
            P(f"p{page.page_number}: {line[:80]}")

P("\n=== question_type marker lines (first 30) ===")
cnt = 0
for page in pages:
    for raw in page.text.splitlines():
        line = raw.strip()
        if not line: continue
        qt = parser._question_type(line)
        if qt:
            P(f"p{page.page_number}: {line[:80]} -> {qt}")
            cnt += 1
            if cnt >= 30: break
    if cnt >= 30: break

# Distribution of question_type
P("\n=== question.question_type distribution ===")
P(Counter(q.question_type for q in questions))
P("\n=== answer.question_type distribution ===")
P(Counter(a.question_type for a in answers))

P("\n=== question chapter_key distribution (top 10) ===")
P(Counter(ckey(q.chapter) for q in questions).most_common(10))
P("\n=== answer chapter_key distribution (top 10) ===")
P(Counter(ckey(a.chapter) for a in answers).most_common(10))

answer_map = {(ckey(a.chapter), a.question_type, a.number): a for a in answers}
from collections import defaultdict
fallback = defaultdict(list)
for a in answers:
    fallback[(ckey(a.chapter), a.number)].append(a)

unmatched_q = []
matched = 0
for q in questions:
    k = (ckey(q.chapter), q.question_type, q.number)
    a = answer_map.pop(k, None)
    if a is None:
        cand = fallback.get((ckey(q.chapter), q.number), [])
        if len(cand) == 1:
            a = cand[0]
    if a is None:
        unmatched_q.append(q)
    else:
        matched += 1

unmatched_answers_left = list(answer_map.values())
P(f"\nmatched: {matched}, unmatched_questions: {len(unmatched_q)}, unmatched_answers_left: {len(unmatched_answers_left)}")

P("\n=== unmatched answers left: distribution by (chapter_key, question_type) (top 15) ===")
P(Counter((ckey(a.chapter), a.question_type) for a in unmatched_answers_left).most_common(15))

P("\n=== sample unmatched answers (first 12) ===")
for a in unmatched_answers_left[:12]:
    P(f"[chap={a.chapter!r} type={a.question_type!r} num={a.number!r} page={a.page}] {' '.join(a.answer_lines)[:130]}")

P("\n=== sample questions (first 12) ===")
for q in questions[:12]:
    P(f"[chap={q.chapter!r} type={q.question_type!r} num={q.number!r} page={q.page}] {' '.join(q.text_lines)[:100]}")

P("\n=== question num ranges per chapter_key (first 8) ===")
ch_nums = defaultdict(list)
for q in questions:
    ch_nums[ckey(q.chapter)].append(int(q.number) if q.number.isdigit() else -1)
for ck, nums in list(ch_nums.items())[:8]:
    P(f"{ck}: min={min(nums)} max={max(nums)} count={len(nums)}")

P("\n=== answer num ranges per chapter_key (first 8) ===")
ch_ans = defaultdict(list)
for a in answers:
    ch_ans[ckey(a.chapter)].append(int(a.number) if a.number.isdigit() else -1)
for ck, nums in list(ch_ans.items())[:8]:
    P(f"{ck}: min={min(nums)} max={max(nums)} count={len(nums)}")

# Why so few questions? Count lines in question section that look like "N. text"
P("\n=== DEBUG: lines matching _QUESTION_RE while section=question ===")
qmatch_count = 0
qmatch_after_type = 0  # match but preceded by a type marker (would be captured)
sample_missed = []
sec = "question"
prev_was_type = False
for page in pages:
    for raw in page.text.splitlines():
        line = raw.strip()
        if not line: continue
        if parser._is_answer_marker(line):
            sec = "answer"; prev_was_type=False; continue
        if parser._is_question_section_marker(line):
            sec = "question"; prev_was_type=False; continue
        qt = parser._question_type(line)
        if qt:
            prev_was_type = True
            if sec != "answer": sec = "question"
            continue
        m = ep._QUESTION_RE.match(line)
        if m and sec == "question":
            qmatch_count += 1
            if len(sample_missed) < 25:
                sample_missed.append(f"p{page.page_number} num={m.group(1)}: {line[:90]}")
        prev_was_type = False
P(f"total _QUESTION_RE matches in question section: {qmatch_count}")
P(f"sample of these (these ARE being captured as questions, num may repeat across types):")
for s in sample_missed:
    P(s)

# How many distinct (chapter, type, number) in questions vs answers?
P("\n=== distinct question keys vs answer keys ===")
qkeys = set((q.chapter, q.question_type, q.number) for q in questions)
P(f"distinct question (chapter,type,num): {len(qkeys)}")
akeys = set((a.chapter, a.question_type, a.number) for a in answers)
P(f"distinct answer (chapter,type,num): {len(akeys)}")

P("\n=== DEBUG: section transitions page-by-page (first marker per page) ===")
sec = "question"
trans = []
for page in pages:
    first = None
    for raw in page.text.splitlines():
        line = raw.strip()
        if not line: continue
        if parser._is_answer_marker(line):
            if sec != "answer":
                trans.append(f"p{page.page_number} question->answer: {line[:50]}")
            sec = "answer"; break
        if parser._is_question_section_marker(line):
            if sec != "question":
                trans.append(f"p{page.page_number} answer->question: {line[:50]}")
            sec = "question"; break
    else:
        pass
for t in trans[:60]:
    P(t)
P(f"total transitions: {len(trans)}")

P("\n=== page 14 FULL ===")

# Dump the sequence of (page, marker-type) for question_type markers AND
# answer markers, in order — to see the type sequence inside answer blocks
# and verify whether "名词解释" reliably starts a new chapter's question block.
P("\n=== ordered (page, kind, value) for answer/type markers ===")
seq = []
sec = "question"
for page in pages:
    for raw in page.text.splitlines():
        line = raw.strip()
        if not line: continue
        if parser._is_answer_marker(line):
            seq.append((page.page_number, "ANSWER", line[:30])); sec = "answer"; continue
        if parser._is_question_section_marker(line):
            seq.append((page.page_number, "QSEC", line[:30])); sec = "question"; continue
        qt = parser._question_type(line)
        if qt:
            seq.append((page.page_number, sec, qt))
for i, (pg, k, v) in enumerate(seq[:140]):
    P(f"{i:3d} p{pg} [{k}] {v}")

# Show first 3 non-empty lines of each page 13-30 to see header pattern
P("\n=== first-3-lines per page (p13-p40) ===")
for i in range(12, 40):
    if i >= len(pages): break
    lines = [l.strip() for l in pages[i].text.splitlines() if l.strip()][:3]
    P(f"-- p{i+1} --")
    for l in lines:
        P(f"   {l[:70]}")

P("\n=== page 38 STRIPPED ===")
cl = ep._strip_repeated_headers(pages)[37]
P("\n".join(cl[:40]))
P("\n=== page 39 STRIPPED ===")
cl = ep._strip_repeated_headers(pages)[38]
P("\n".join(cl[:40]))

out.write_text("\n".join(buf), encoding="utf-8")
print(f"WRITTEN: {out}")