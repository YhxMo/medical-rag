"""PDF text loading with OCR-candidate detection."""
from __future__ import annotations

from pathlib import Path

from src.schema import PageDocument


class PDFLoader:
    """Load PDF pages with PyMuPDF and mark OCR candidates early."""

    def __init__(self, file_path: str | Path, *, ocr_min_chars: int = 50) -> None:
        self.file_path = Path(file_path)
        self.ocr_min_chars = ocr_min_chars

    def load(self) -> list[PageDocument]:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required to load PDF files.") from exc

        documents: list[PageDocument] = []
        with fitz.open(self.file_path) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                text = page.get_text("text")
                documents.append(
                    PageDocument.from_text(
                        source_file=self.file_path.name,
                        page_number=page_index,
                        text=text,
                        ocr_min_chars=self.ocr_min_chars,
                        metadata={"path": str(self.file_path)},
                    )
                )
        return documents
