"""PDF -> page-local text chunks + optional figure descriptions -> one index."""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.application.images import prepare_image
from src.application.prompts import CAPTION
from src.document.chunker import chunk_pages
from src.document.loader import load_pdf
from src.schema import EvidenceItem


def ingest(data_dir, store, embedding, *, chunk_size=900, overlap=100, vision=None):
    pdfs = sorted(p for p in Path(data_dir).glob("*.pdf") if not p.name.startswith("."))
    if not pdfs:
        raise ValueError(f"No PDF files found in {data_dir}")
    if vision is not None and not vision.available:
        raise ValueError("Configure the vision API key before using --captions")
    evidence = []
    for path in pdfs:
        pages = load_pdf(path)
        evidence.extend(chunk_pages(pages, chunk_size=chunk_size, overlap=overlap))
        if vision is not None:
            evidence.extend(caption_pages(path, pages, vision, store.index_dir.parent / "images"))
    if not evidence:
        raise ValueError(
            "PDFs contain no extractable text. Use text PDFs or enable --captions for scanned pages."
        )
    # Model/PDF failures occur before replacing a working index.
    store.build(evidence, embedding)
    return len(evidence)


def caption_pages(path, pages, vision, image_dir):
    """One visual description per page containing raster images (including scans)."""
    import pymupdf

    image_dir.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    with pymupdf.open(path) as pdf:
        for page, document in zip(pdf, pages, strict=True):
            if not page.get_images():
                continue
            image = image_dir / f"{source_hash}-p{document.page_number}.png"
            page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(image)
            caption = vision.generate(
                CAPTION, {"nearby_text": document.text[:3000]}, images=[prepare_image(image)]
            )
            yield EvidenceItem(
                f"{source_hash}:figure:{document.page_number}",
                "image_caption",
                path.name,
                caption,
                document.page_number,
                document.page_number,
                {"image_path": str(image.resolve()), "derived_by_model": True},
            )
