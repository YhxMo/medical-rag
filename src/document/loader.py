"""Extract PDF text with source filenames and one-based PDF page numbers."""

from pathlib import Path

import pymupdf

from src.schema import PageDocument


def load_pdf(file_path: str | Path) -> list[PageDocument]:
    path = Path(file_path)
    with pymupdf.open(path) as pdf:
        return [
            PageDocument.from_text(path.name, number, page.get_text("text"))
            for number, page in enumerate(pdf, start=1)
        ]
