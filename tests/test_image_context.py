from pathlib import Path

from src.indexer.image_context import (
    attach_context_to_caption,
    build_contextual_caption_prompt,
    build_image_caption_context,
)
from src.schema import ImageAsset, ImageCaption, PageDocument, TextChunk


def test_build_image_caption_context_collects_heading_caption_and_nearby_text():
    asset = ImageAsset("img1", "book.pdf", 3, Path("page.png"), kind="embedded")
    pages = [
        PageDocument.from_text(
            "book.pdf",
            3,
            "第二章 中枢神经系统\n图2-1 急性脑出血CT表现\n脑出血CT常表现为高密度影。",
        )
    ]
    chunks = [
        TextChunk(
            chunk_id="c1",
            source_file="book.pdf",
            page_start=3,
            page_end=3,
            heading_path=("第二章 中枢神经系统", "一、脑血管疾病"),
            content="脑出血CT常表现为高密度血肿，周围可见水肿。",
            chunk_strategy="layout_heading",
        )
    ]

    context = build_image_caption_context(asset, pages=pages, text_chunks=chunks)

    assert context["heading_path"] == ["第二章 中枢神经系统", "一、脑血管疾病"]
    assert context["figure_captions"] == ["图2-1 急性脑出血CT表现"]
    assert "高密度血肿" in context["nearby_text"]


def test_contextual_caption_prompt_and_metadata_include_source_context():
    caption = ImageCaption("cap1", "img1", "book.pdf", 3, "头颅CT显示高密度灶。")
    context = {
        "source_file": "book.pdf",
        "page_number": 3,
        "image_kind": "page",
        "heading_path": ["第二章"],
        "figure_captions": ["图2-1 脑出血"],
        "nearby_text": "急性脑出血CT表现为高密度。",
    }

    prompt = build_contextual_caption_prompt("请描述图片。", context)
    enriched = attach_context_to_caption(caption, context)

    assert "图注候选：图2-1 脑出血" in prompt
    assert "附近正文：急性脑出血CT表现为高密度。" in prompt
    assert enriched.metadata["heading_path"] == ["第二章"]
