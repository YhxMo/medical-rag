from src.evaluation.generator import generate_draft_questions
from src.schema import EvidenceItem


def test_generate_draft_questions_filters_low_signal_content():
    low_signal = EvidenceItem(
        "e1",
        "text",
        "book.pdf",
        "15\n目录\n绪论\b1\n一、  医学影像学发展简史\b����������",
    )
    real_content = EvidenceItem(
        "e2",
        "text",
        "book.pdf",
        "脑出血在CT上通常表现为边界清楚的高密度影，血肿周围可见低密度水肿带，"
        "急性期密度较均匀，随时间推移密度逐渐减低并液化。",
    )

    questions = generate_draft_questions([low_signal, real_content], limit=30)

    expected_ids = {question.expected_evidence_ids[0] for question in questions}
    assert "e2" in expected_ids
    assert "e1" not in expected_ids


def test_generate_draft_questions_samples_evenly_across_full_list():
    evidence = [
        EvidenceItem(
            f"e{index}",
            "text",
            "book.pdf",
            f"脑出血在CT上的典型表现为边界清楚的高密度影，血肿周围可见低密度水肿带，"
            f"此为第{index}段教材正文内容，用于验证均匀采样是否覆盖全书而非仅集中在开头。",
        )
        for index in range(100)
    ]

    questions = generate_draft_questions(evidence, limit=10)

    sampled_ids = [question.expected_evidence_ids[0] for question in questions]
    assert len(sampled_ids) == 10
    # Evenly spread sampling should reach well past the first ~10 items instead of
    # clustering at the head of the evidence list.
    numeric_indices = [int(item_id.removeprefix("e")) for item_id in sampled_ids]
    assert max(numeric_indices) >= 80


def test_generate_draft_questions_keeps_exercise_qa_regardless_of_shape():
    exercise = EvidenceItem(
        "q1",
        "exercise_qa",
        "exercises.pdf",
        "题目：1\n答案：A",
        metadata={"question_text": "脑出血的CT表现是什么？"},
    )

    questions = generate_draft_questions([exercise], limit=30)

    assert len(questions) == 1
    assert questions[0].question == "脑出血的CT表现是什么？"


def test_generate_draft_questions_filters_reference_list_pages():
    reference_list = EvidenceItem(
        "e1",
        "text",
        "book.pdf",
        "推荐阅读\n372\n［1］\t龚启勇，冯晓源．神经放射诊断学．北京：人民卫生出版社，2018．\n"
        "［2］\tGONG QY, et al. Neuroradiology diagnostics. Beijing, 2018.",
    )
    real_content = EvidenceItem(
        "e2",
        "text",
        "book.pdf",
        "脑出血在CT上通常表现为边界清楚的高密度影，血肿周围可见低密度水肿带，"
        "急性期密度较均匀，随时间推移密度逐渐减低并液化。",
    )

    questions = generate_draft_questions([reference_list, real_content], limit=30)

    expected_ids = {question.expected_evidence_ids[0] for question in questions}
    assert "e2" in expected_ids
    assert "e1" not in expected_ids


def test_generate_draft_questions_strips_leading_navigation_noise():
    navigation_prefixed = EvidenceItem(
        "e1",
        "text",
        "book.pdf",
        "本章数字资源\n本章思维导图\n脑出血在CT上通常表现为边界清楚的高密度影，血肿周围可见低密度水肿带，"
        "急性期密度较均匀，随时间推移密度逐渐减低并液化。",
    )

    questions = generate_draft_questions([navigation_prefixed], limit=30)

    assert len(questions) == 1
    assert "本章数字资源" not in questions[0].question
    assert "本章思维导图" not in questions[0].question
    assert "脑出血" in questions[0].question


def test_generate_draft_questions_strips_navigation_noise_after_leading_page_number():
    # Real-world shape: a bare page-number line precedes the navigation markers
    # (e.g. "127\n本章思维导图\n本章数字资源\n第五章 \n循环系统...").
    page_number_then_navigation = EvidenceItem(
        "e1",
        "text",
        "book.pdf",
        "127\n本章思维导图\n本章数字资源\n第五章 \n循环系统\n人体的循环系统，由心脏和全身的血管系统组成，"
        "心脏作为中心器官，具有泵血功能，推动血液在血管中循环流动。",
    )

    questions = generate_draft_questions([page_number_then_navigation], limit=30)

    assert len(questions) == 1
    assert "本章思维导图" not in questions[0].question
    assert "本章数字资源" not in questions[0].question
    assert "循环系统" in questions[0].question
