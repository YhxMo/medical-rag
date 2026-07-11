from src.indexer.evidence import build_evidence_items
from src.schema import ExerciseQAChunk, ImageCaption, TextChunk


def test_build_evidence_items_keeps_text_chunk_strategy():
    text_chunk = TextChunk(
        chunk_id="c1",
        source_file="book.pdf",
        page_start=1,
        page_end=2,
        heading_path=("第一章",),
        content="脑出血CT表现。",
        chunk_strategy="layout_heading",
    )

    evidence = build_evidence_items(text_chunks=[text_chunk])

    assert evidence[0].evidence_type == "text"
    assert evidence[0].metadata["chunk_strategy"] == "layout_heading"


def test_build_evidence_items_keeps_image_caption_metadata():
    caption = ImageCaption(
        caption_id="cap1",
        image_id="img1",
        source_file="book.pdf",
        page_number=5,
        caption="一张CT图像。",
        is_ocr_text=True,
        metadata={
            "heading_path": ["第二章 中枢神经系统"],
            "figure_captions": ["图2-1 脑出血CT"],
            "nearby_text": "脑出血CT表现为高密度。",
        },
    )

    evidence = build_evidence_items(captions=[caption])

    assert evidence[0].evidence_type == "image_caption"
    assert evidence[0].metadata["is_ocr_text"] is True
    assert evidence[0].page_start == 5
    assert "章节：第二章 中枢神经系统" in evidence[0].content
    assert "图注：图2-1 脑出血CT" in evidence[0].content
    assert "附近正文：脑出血CT表现为高密度。" in evidence[0].content


def test_build_evidence_items_formats_exercise_as_one_chunk():
    exercise = ExerciseQAChunk(
        question_id="q1",
        source_file="习题.pdf",
        chapter="第一章",
        question_number="1",
        question_type="选择题",
        question_text="CT值单位是？",
        options=("A. HU", "B. mmHg"),
        answer="A",
        explanation="CT值单位为HU。",
        page_question=10,
        page_answer=80,
    )

    evidence = build_evidence_items(exercise_chunks=[exercise])

    assert evidence[0].evidence_id == "q1"
    assert evidence[0].evidence_type == "exercise_qa"
    assert "题目" in evidence[0].content
    assert evidence[0].metadata["chunk_strategy"] == "exercise_qa_pair"
