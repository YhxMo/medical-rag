"""PDF文档解析与向量知识库构建"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    DATA_DIR, VECTOR_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EXCLUDED_PDF_NAMES,
    EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE, EMBEDDING_LOCAL_FILES_ONLY,
)

logger = logging.getLogger(__name__)


def get_embedding_model_kwargs() -> dict[str, object]:
    """返回SentenceTransformer初始化参数。"""
    return {
        "device": EMBEDDING_DEVICE,
        "local_files_only": EMBEDDING_LOCAL_FILES_ONLY,
    }


def get_source_pdf_files(
    pdf_dir: str | Path = DATA_DIR,
    excluded_pdf_names: set[str] | None = None,
) -> list[Path]:
    """返回可进入正式知识库的PDF文件列表。"""
    excluded = EXCLUDED_PDF_NAMES if excluded_pdf_names is None else excluded_pdf_names
    pdf_dir = Path(pdf_dir)
    return [
        pdf_path
        for pdf_path in sorted(pdf_dir.glob("*.pdf"))
        if pdf_path.name not in excluded
    ]


def reset_vector_store_dir(persist_dir: str | Path = VECTOR_DB_DIR) -> None:
    """清空并重建Chroma持久化目录，避免重复索引旧数据。"""
    persist_path = Path(persist_dir)
    if persist_path.exists():
        if not persist_path.is_dir():
            raise ValueError(f"向量库路径不是目录: {persist_path}")
        shutil.rmtree(persist_path)
    persist_path.mkdir(parents=True, exist_ok=True)


def parse_pdf_text(pdf_path: str | Path) -> str:
    """从PDF文件中提取纯文本。

    Args:
        pdf_path: PDF文件路径

    Returns:
        提取的完整文本

    Raises:
        FileNotFoundError: PDF文件不存在
        ValueError: PDF无法解析或文本为空
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    text_parts: list[str] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        raise ValueError(f"PDF解析失败 ({pdf_path}): {e}") from e

    full_text = "\n".join(text_parts).strip()
    if not full_text:
        raise ValueError(f"PDF文本为空: {pdf_path}")

    logger.info("从 %s 解析出 %d 字符", pdf_path.name, len(full_text))
    return full_text


def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """将长文本分割为适合嵌入的chunk。

    Args:
        text: 输入文本
        chunk_size: 每个chunk的最大字符数
        chunk_overlap: 相邻chunk的重叠字符数

    Returns:
        chunk字符串列表，空输入返回空列表
    """
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    logger.info("文本被分割为 %d 个chunk（chunk_size=%d, overlap=%d）",
                len(chunks), chunk_size, chunk_overlap)
    return chunks


def build_vector_store(
    pdf_dir: str | Path = DATA_DIR,
    persist_dir: str | Path = VECTOR_DB_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Chroma:
    """解析目录下所有PDF，构建ChromaDB向量知识库。

    Args:
        pdf_dir: 包含PDF文件的目录
        persist_dir: ChromaDB持久化路径
        chunk_size: 分块大小
        chunk_overlap: 分块重叠

    Returns:
        构建好的Chroma向量存储实例

    Raises:
        FileNotFoundError: 目录下无PDF文件
    """
    pdf_dir = Path(pdf_dir)
    pdf_files = get_source_pdf_files(pdf_dir)
    if not pdf_files:
        raise FileNotFoundError(f"目录下未找到PDF文件: {pdf_dir}")

    all_documents: list[Document] = []
    for pdf_path in pdf_files:
        logger.info("处理PDF: %s", pdf_path.name)
        text = parse_pdf_text(pdf_path)
        chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": pdf_path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            all_documents.append(doc)

    logger.info("共生成 %d 个Document，开始向量化...", len(all_documents))

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=get_embedding_model_kwargs(),
        encode_kwargs={"normalize_embeddings": True},
    )

    reset_vector_store_dir(persist_dir)

    vector_store = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )

    logger.info("向量知识库构建完成，共 %d 条记录，持久化到 %s",
                len(all_documents), persist_dir)
    return vector_store


def load_vector_store(persist_dir: str | Path = VECTOR_DB_DIR) -> Chroma:
    """加载已有的ChromaDB向量知识库。

    Args:
        persist_dir: ChromaDB持久化路径

    Returns:
        Chroma向量存储实例
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=get_embedding_model_kwargs(),
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    # NOTE: ChromaDB does not expose a public count() method on the vector store.
    # We use .get() which returns all ids; for large collections this may be slow.
    collection_data = vector_store.get()
    record_count = len(collection_data.get("ids", [])) if collection_data else 0
    logger.info("向量知识库加载完成，共 %d 条记录", record_count)
    return vector_store


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_vector_store()
