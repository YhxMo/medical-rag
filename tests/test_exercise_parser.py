from pathlib import Path

from src.document.exercise_parser import ExerciseParser
from src.schema import PageDocument
from scripts.report_exercise_parse import build_book_stats, render_markdown_report


def test_exercise_parser_pairs_questions_and_answers_by_chapter_and_number():
    pages = [
        PageDocument.from_text(
            "医学影像学习题集.pdf",
            10,
            "\n".join(
                [
                    "第一章 总论",
                    "选择题",
                    "1. X线成像的基础是？",
                    "A. 密度差异",
                    "B. 温度差异",
                    "2. CT值的单位是？",
                    "A. HU",
                    "B. mmHg",
                ]
            ),
        ),
        PageDocument.from_text(
            "医学影像学习题集.pdf",
            80,
            "\n".join(
                [
                    "第一章 总论",
                    "参考答案与解析",
                    "1. A",
                    "解析 X线成像依赖组织密度和厚度差异。",
                    "2. A",
                    "解析 CT值单位为HU。",
                ]
            ),
        ),
    ]

    report = ExerciseParser().parse(pages)

    assert len(report.chunks) == 2
    assert report.chunks[0].question_id == "医学影像学习题集:第一章 总论:1"
    assert report.chunks[0].answer == "A"
    assert "密度" in report.chunks[0].explanation
    assert report.chunks[0].page_question == 10
    assert report.chunks[0].page_answer == 80
    assert not report.unmatched_questions
    assert not report.unmatched_answers


def test_exercise_parser_keeps_unmatched_questions():
    pages = [
        PageDocument.from_text("习题.pdf", 1, "第一章 总论\n选择题\n1. 未配对题目？"),
    ]

    report = ExerciseParser().parse(pages)

    assert len(report.chunks) == 1
    assert len(report.unmatched_questions) == 1
    assert report.chunks[0].answer == ""


def test_exercise_parser_pairs_compact_answer_lines_by_question_type():
    pages = [
        PageDocument.from_text(
            "习题.pdf",
            1,
            "\n".join(
                [
                    "第一章 影像诊断学总论",
                    "【A1型题】",
                    "1. CT值的单位是？",
                    "A. HU",
                    "B. mmHg",
                    "2. MRI图像灰度称为什么？",
                    "A. 密度",
                    "B. 信号强度",
                    "四、参考答案",
                    "【A1型题】",
                    "1.A2.B",
                ]
            ),
        )
    ]

    report = ExerciseParser().parse(pages)

    assert len(report.chunks) == 2
    assert report.chunks[0].answer == "A"
    assert report.chunks[1].answer == "B"
    assert not report.unmatched_questions


def test_exercise_parser_disambiguates_duplicate_numbers_across_types():
    pages = [
        PageDocument.from_text(
            "习题.pdf",
            1,
            "\n".join(
                [
                    "第一章 总论",
                    "选择题",
                    "1. CT值单位是？",
                    "A. HU",
                    "B. mmHg",
                    "简答题",
                    "1. 简述MRI图像特点。",
                    "参考答案",
                    "选择题",
                    "1.A",
                    "简答题",
                    "1.MRI为多序列、多参数成像。",
                ]
            ),
        )
    ]

    report = ExerciseParser().parse(pages)

    assert [chunk.answer for chunk in report.chunks] == ["A", "MRI为多序列、多参数成像。"]
    assert report.chunks[0].question_id == "习题:第一章 总论:选择题:1"
    assert report.chunks[1].question_id == "习题:第一章 总论:简答题:1"


def test_exercise_parser_disambiguates_repeated_numbers_with_stable_occurrence():
    pages = [
        PageDocument.from_text(
            "习题.pdf",
            1,
            "\n".join(
                [
                    "第八章 泌尿生殖系统",
                    "病例分析",
                    "1. 该患者首先考虑哪种疾病？",
                    "2. 该疾病主要鉴别诊断有哪些？",
                    "1. 病变的位置属于哪个间隙？考虑为哪种疾病？",
                    "2. 病变有哪些CT影像学表现特点？",
                ]
            ),
        )
    ]

    report = ExerciseParser().parse(pages)

    assert [chunk.question_number for chunk in report.chunks] == ["1", "2", "1", "2"]
    assert [chunk.question_id for chunk in report.chunks] == [
        "习题:第八章 泌尿生殖系统:病例分析:1:occurrence1",
        "习题:第八章 泌尿生殖系统:病例分析:2:occurrence1",
        "习题:第八章 泌尿生殖系统:病例分析:1:occurrence2",
        "习题:第八章 泌尿生殖系统:病例分析:2:occurrence2",
    ]
    assert [chunk.metadata["question_id_occurrence"] for chunk in report.chunks] == [1, 1, 2, 2]


def test_exercise_parser_disambiguates_repeated_case_subquestions_with_stable_occurrence():
    pages = [
        PageDocument.from_text(
            "习题.pdf",
            1,
            "\n".join(
                [
                    "第八章 泌尿生殖系统",
                    "附录病例分析",
                    "【病例8-6】",
                    "患者A，行CT检查。",
                    "【提问】",
                    "1. 该病例所见应诊断为哪种疾病？",
                    "2. 需进一步进行哪些检查？",
                    "【病例8-6】",
                    "患者B，行MR检查。",
                    "【提问】",
                    "1. 该患者首先考虑为哪种疾病？",
                    "2. 该疾病影像学检查的目的是什么？",
                ]
            ),
        )
    ]

    report = ExerciseParser().parse(pages)

    case_chunks = [chunk for chunk in report.chunks if chunk.question_type == "病例分析"]
    assert [chunk.question_number for chunk in case_chunks] == ["8-6:1", "8-6:2", "8-6:1", "8-6:2"]
    assert [chunk.question_id for chunk in case_chunks] == [
        "习题:第八章 泌尿生殖系统:病例分析:8-6:1:occurrence1",
        "习题:第八章 泌尿生殖系统:病例分析:8-6:2:occurrence1",
        "习题:第八章 泌尿生殖系统:病例分析:8-6:1:occurrence2",
        "习题:第八章 泌尿生殖系统:病例分析:8-6:2:occurrence2",
    ]
    assert "患者A" in case_chunks[0].question_text
    assert "患者B" in case_chunks[2].question_text


def test_exercise_parser_returns_to_question_section_after_toc_answer_marker():
    pages = [
        PageDocument.from_text(
            "习题.pdf",
            1,
            "目录\n三、习题\n1\n四、参考答案\n2",
        ),
        PageDocument.from_text(
            "习题.pdf",
            2,
            "\n".join(
                [
                    "第一章 绪论",
                    "三、习题",
                    "（一）名词解释",
                    "1. 同病异影",
                    "四、参考答案",
                    "（一）名词解释",
                    "1. 同一疾病不同阶段影像表现不同。",
                ]
            ),
        ),
    ]

    report = ExerciseParser().parse(pages)

    assert len(report.chunks) == 1
    assert report.chunks[0].question_text == "同病异影"
    assert "不同阶段" in report.chunks[0].answer


def test_exercise_parser_matches_split_and_full_chapter_titles_by_chapter_number():
    pages = [
        PageDocument.from_text(
            "习题.pdf",
            1,
            "\n".join(
                [
                    "第一章",
                    "影像诊断学总论",
                    "二、复习思考题",
                    "（一）名词解释",
                    "1. 数字化X线成像",
                    "参考答案",
                    "第一章影像诊断学总论",
                    "（一）名词解释",
                    "1. 数字化X线成像：将X线信息数字化处理并显示。",
                ]
            ),
        )
    ]

    report = ExerciseParser().parse(pages)

    assert len(report.chunks) == 1
    assert "数字化处理" in report.chunks[0].answer


def test_exercise_parser_pairs_case_analysis_sub_questions_by_case_id():
    pages = [
        PageDocument.from_text(
            "习题.pdf",
            1,
            "\n".join(
                [
                    "第二章 中枢神经系统",
                    "附录病例分析",
                    "【病例2-1】",
                    "患者，女性，56岁，书写障碍伴视力下降20天，行颅脑MRI检查。",
                    "【提问】",
                    "1.该病主要影像学表现是什么？",
                    "2.该病的诊断及鉴别诊断是什么？",
                    "参考答案",
                    "【病例2-1】",
                    "1.影像学表现左侧顶叶见单发团片状异常信号影。",
                    "2.诊断及鉴别诊断诊断：左侧顶叶胶质母细胞瘤。",
                ]
            ),
        )
    ]

    report = ExerciseParser().parse(pages)

    case_chunks = [c for c in report.chunks if c.question_type == "病例分析"]
    assert len(case_chunks) == 2
    # Sub-question numbers are namespaced under the case id so they stay unique.
    assert {c.question_number for c in case_chunks} == {"2-1:1", "2-1:2"}
    # Each sub-question carries the case stem as context, then its own prompt.
    assert "患者，女性" in case_chunks[0].question_text
    assert "影像学表现是什么" in case_chunks[0].question_text
    # Sub-questions pair with the identically-numberd sub-answers.
    by_num = {c.question_number: c for c in case_chunks}
    assert "左侧顶叶见单发团片状异常信号影" in by_num["2-1:1"].answer
    assert "胶质母细胞瘤" in by_num["2-1:2"].answer
    assert not report.unmatched_questions
    assert not report.unmatched_answers


def test_exercise_parse_report_summarizes_counts_and_samples():
    pages = [
        PageDocument.from_text(
            "医学影像学习题集.pdf",
            1,
            "\n".join(
                [
                    "第一章 总论",
                    "选择题",
                    "1. CT值单位是？",
                    "2. 未配对题目？",
                    "参考答案",
                    "选择题",
                    "1. HU",
                    "3. 多余答案",
                ]
            ),
        )
    ]
    report = ExerciseParser().parse(pages)

    stats = build_book_stats(Path("data/医学影像学习题集.pdf"), report)
    markdown = render_markdown_report([stats])

    assert stats.chunks == 2
    assert stats.unmatched_questions == 1
    assert stats.unmatched_answers == 1
    assert stats.estimated_pairing_rate == 1 / 3
    assert ("第一章 总论", 2) in stats.by_chapter
    assert ("选择题", 2) in stats.by_question_type
    assert "| 医学影像学习题集.pdf | 2 | 1 | 1 | 33.3% |" in markdown
    assert "未配对题目" in markdown
