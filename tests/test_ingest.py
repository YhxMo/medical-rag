"""测试文档解析与分块功能"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import EXCLUDED_PDF_NAMES
from ingest import (
    Chroma,
    HuggingFaceEmbeddings,
    build_vector_store,
    get_source_pdf_files,
    load_vector_store,
    parse_pdf_text,
    split_text_into_chunks,
)


SAMPLE_TEXT = (
    "第一章 胸部CT基础\n"
    "胸部CT是临床常用影像学检查方法。肺窗主要用于观察肺实质病变，"
    "纵隔窗则用于评估纵隔结构及胸壁软组织。\n"
    "第二章 肺结节影像学表现\n"
    "肺磨玻璃结节（GGN）是指CT上表现为局灶性密度增高但不掩盖血管"
    "和支气管走行的病变区域。根据是否含有实性成分，可分为纯磨玻璃"
    "结节和部分实性结节。\n"
    "肺结节的随访管理需参照Fleischner学会指南，根据结节大小及风险"
    "因素制定个体化的随访方案。\n"
    "第三章 肺炎的影像学诊断\n"
    "肺炎在胸部CT上可表现为肺实变、磨玻璃影、小叶中心结节等多种"
    "征象。细菌性肺炎常表现为大叶性实变，而病毒性肺炎则以磨玻璃影"
    "和间质增厚为主要特征。影像学检查对于肺炎的早期发现、病情评估"
    "和治疗效果监测具有重要价值。\n"
    "第四章 肺癌的CT筛查\n"
    "低剂量螺旋CT是目前国际公认的肺癌筛查金标准。肺结节的大小、"
    "密度、边缘特征及生长速度是评估恶性风险的关键指标。人工智能"
    "辅助诊断系统在肺结节检测和分类方面展现出良好的应用前景，可以"
    "提高早期肺癌的检出率并降低假阳性结果。"
)


def test_split_text_into_chunks_returns_correct_number():
    """分块数量应随chunk_size正确变化"""
    chunks = split_text_into_chunks(SAMPLE_TEXT, chunk_size=200, chunk_overlap=50)

    assert len(chunks) >= 3, f"期望至少3个chunk，实际得到{len(chunks)}"
    # 验证每个chunk长度不超过chunk_size
    for chunk in chunks:
        assert len(chunk) <= 200


def test_split_text_into_chunks_preserves_content():
    """分块不应丢失关键信息"""
    chunks = split_text_into_chunks(SAMPLE_TEXT, chunk_size=300, chunk_overlap=50)

    combined = "".join(chunks)
    # 核心术语应出现在合并后的文本中
    for keyword in ["肺磨玻璃结节", "GGN", "Fleischner", "纵隔窗"]:
        assert keyword in combined, f"关键词'{keyword}'在分块后丢失"


def test_split_text_into_chunks_empty_input():
    """空文本应返回空列表"""
    chunks = split_text_into_chunks("", chunk_size=200, chunk_overlap=50)

    assert chunks == []


def test_split_text_into_chunks_short_text():
    """短于chunk_size的文本应返回包含全文的单个chunk"""
    short = "胸部CT是常用的检查方法。"
    chunks = split_text_into_chunks(short, chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 1
    assert chunks[0] == short


def test_parse_pdf_text_file_not_found():
    """不存在的PDF文件应抛出FileNotFoundError"""
    with pytest.raises(FileNotFoundError, match="PDF文件不存在"):
        parse_pdf_text("nonexistent_file_12345.pdf")


def test_parse_pdf_text_empty_pdf():
    """空PDF应抛出ValueError"""
    import io
    import tempfile
    from pathlib import Path

    # 创建一个最小但有效的空PDF
    empty_pdf = io.BytesIO()
    empty_pdf.write(
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
        b"xref\n"
        b"0 3\n"
        b"0000000000 65535 f \n"
        b"trailer<</Size 3/Root 1 0 R>>\n"
        b"startxref\n"
        b"%%EOF\n"
    )
    empty_pdf.seek(0)

    # 写入临时文件
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(empty_pdf.read())
        tmp_path = f.name

    try:
        with pytest.raises(ValueError, match="PDF文本为空"):
            parse_pdf_text(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_get_source_pdf_files_excludes_known_risky_pdf(tmp_path):
    """已知文本抽取有风险的PDF不应进入正式知识库。"""
    risky_name = "Chinese_Pulmonary_Nodule_Consensus_2018_Interpretation.pdf"
    trusted_pdf = tmp_path / "trusted_guideline.pdf"
    risky_pdf = tmp_path / risky_name
    trusted_pdf.write_bytes(b"%PDF-1.4 trusted")
    risky_pdf.write_bytes(b"%PDF-1.4 risky")

    source_files = get_source_pdf_files(tmp_path)

    assert risky_name in EXCLUDED_PDF_NAMES
    assert trusted_pdf in source_files
    assert risky_pdf not in source_files


def test_vector_store_integrations_use_langchain_1x_packages():
    """向量库集成应使用LangChain 1.x推荐的独立包。"""
    assert HuggingFaceEmbeddings.__module__.startswith("langchain_huggingface")
    assert Chroma.__module__.startswith("langchain_chroma")


@patch("ingest.Chroma")
@patch("ingest.HuggingFaceEmbeddings")
def test_load_vector_store_uses_local_embedding_cache(mock_embeddings, mock_chroma):
    """加载向量库时应避免HuggingFace远程探测导致客户端关闭。"""
    mock_store = MagicMock(name="vector_store")
    mock_store.get.return_value = {"ids": ["doc-1"]}
    mock_chroma.return_value = mock_store

    vector_store = load_vector_store()

    assert vector_store is mock_store
    mock_embeddings.assert_called_once()
    assert mock_embeddings.call_args.kwargs["model_kwargs"]["local_files_only"] is True


@patch("ingest.Chroma")
@patch("ingest.HuggingFaceEmbeddings")
@patch("ingest.parse_pdf_text")
def test_build_vector_store_clears_existing_persist_dir(
    mock_parse_pdf_text,
    mock_embeddings,
    mock_chroma,
    tmp_path,
):
    """重建向量库前应清空旧持久化目录，避免重复索引。"""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "trusted_guideline.pdf").write_bytes(b"%PDF-1.4 trusted")

    persist_dir = tmp_path / "vector_db"
    persist_dir.mkdir()
    stale_file = persist_dir / "stale.sqlite3"
    stale_file.write_text("old vector data", encoding="utf-8")

    mock_parse_pdf_text.return_value = "肺结节随访建议以毫米为单位。"
    mock_embeddings.return_value = MagicMock(name="embeddings")
    expected_store = MagicMock(name="vector_store")
    mock_chroma.from_documents.return_value = expected_store

    vector_store = build_vector_store(
        pdf_dir=pdf_dir,
        persist_dir=persist_dir,
        chunk_size=100,
        chunk_overlap=0,
    )

    assert vector_store is expected_store
    assert not stale_file.exists()
    assert persist_dir.exists()
    mock_chroma.from_documents.assert_called_once()
    indexed_docs = mock_chroma.from_documents.call_args.kwargs["documents"]
    assert [doc.metadata["source"] for doc in indexed_docs] == ["trusted_guideline.pdf"]
