from src.document.chunker import FixedChunker, LayoutHeadingChunker
from src.schema import PageDocument


def test_layout_heading_chunker_preserves_heading_and_metadata():
    pages = [
        PageDocument.from_text(
            "book.pdf",
            1,
            "第一章 总论\n\n脑出血在CT上常表现为高密度影，周围可有水肿带。" * 8,
        )
    ]

    chunks = LayoutHeadingChunker().chunk(pages)

    assert chunks
    assert chunks[0].source_file == "book.pdf"
    assert chunks[0].page_start == 1
    assert chunks[0].heading_path == ("第一章 总论",)
    assert chunks[0].chunk_strategy == "layout_heading"


def test_fixed_chunker_is_available_as_baseline():
    page = PageDocument.from_text("book.pdf", 2, "abcdef" * 200)

    chunks = FixedChunker().chunk([page])

    assert len(chunks) > 1
    assert {chunk.chunk_strategy for chunk in chunks} == {"fixed"}


def test_layout_heading_chunker_keeps_hierarchical_knowledge_path():
    pages = [
        PageDocument.from_text(
            "book.pdf",
            1,
            "\n\n".join(
                [
                    "第一篇 影像诊断学",
                    "第二章 中枢神经系统",
                    "一、脑血管疾病",
                    "（一）脑出血",
                    "急性脑出血CT常表现为高密度血肿，周围可见水肿和占位效应。",
                ]
            ),
        )
    ]

    chunks = LayoutHeadingChunker().chunk(pages)

    assert chunks[0].heading_path == (
        "第一篇 影像诊断学",
        "第二章 中枢神经系统",
        "一、脑血管疾病",
        "（一）脑出血",
    )
