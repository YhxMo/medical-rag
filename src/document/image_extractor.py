"""PDF page and embedded-image extraction."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.schema import ImageAsset, PageDocument


_SAFE_EXTENSIONS = {"png", "jpg", "jpeg"}


@dataclass(frozen=True)
class ImageExtractionConfig:
    output_dir: Path
    render_page_images: bool = True
    extract_embedded_images: bool = True
    min_width: int = 64
    min_height: int = 64
    zoom: float = 2.0


class PDFImageExtractor:
    """Extract page screenshots and embedded images from PDF files."""

    def __init__(self, config: ImageExtractionConfig) -> None:
        self.config = config

    def extract(self, pdf_path: str | Path, pages: list[PageDocument] | None = None) -> list[ImageAsset]:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required to extract PDF images.") from exc

        source = Path(pdf_path)
        output_root = self.config.output_dir / source.stem
        output_root.mkdir(parents=True, exist_ok=True)
        page_by_number = {page.page_number: page for page in pages or []}
        assets: list[ImageAsset] = []

        with fitz.open(source) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                if self.config.render_page_images:
                    assets.append(self._render_page(page, source.name, page_index, output_root))
                if self.config.extract_embedded_images:
                    assets.extend(self._extract_embedded(pdf, page, source.name, page_index, output_root))

                if page_by_number.get(page_index) and page_by_number[page_index].needs_ocr:
                    # The rendered page image above is the OCR fallback artifact.
                    continue
        return assets

    def _render_page(self, page, source_file: str, page_number: int, output_root: Path) -> ImageAsset:
        import fitz

        matrix = fitz.Matrix(self.config.zoom, self.config.zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image_id = _image_id(source_file, page_number, "page", 0)
        path = output_root / f"{image_id}.png"
        pixmap.save(path)
        return ImageAsset(
            image_id=image_id,
            source_file=source_file,
            page_number=page_number,
            path=path,
            kind="page",
            metadata={"width": pixmap.width, "height": pixmap.height},
        )

    def _extract_embedded(self, pdf, page, source_file: str, page_number: int, output_root: Path) -> list[ImageAsset]:
        assets: list[ImageAsset] = []
        for image_index, image_info in enumerate(page.get_images(full=True), start=1):
            xref = image_info[0]
            image = pdf.extract_image(xref)
            width = int(image.get("width", 0))
            height = int(image.get("height", 0))
            if width < self.config.min_width or height < self.config.min_height:
                continue
            extension = image.get("ext", "png")
            image_id = _image_id(source_file, page_number, "embedded", image_index)
            path = output_root / f"{image_id}.{extension}"
            path.write_bytes(image["image"])
            result = _ensure_decodable(pdf, xref, path, extension)
            if result is None:
                # Neither the original bytes nor a re-render could produce a
                # format Pillow/VLM pipelines can open. Skip this one image
                # rather than aborting extraction for the whole document.
                path.unlink(missing_ok=True)
                continue
            path, extension = result
            assets.append(
                ImageAsset(
                    image_id=image_id,
                    source_file=source_file,
                    page_number=page_number,
                    path=path,
                    kind="embedded",
                    metadata={"width": width, "height": height, "xref": xref, "ext": extension},
                )
            )
        return assets


def _image_id(source_file: str, page_number: int, kind: str, index: int) -> str:
    raw = f"{source_file}:{page_number}:{kind}:{index}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"img_{digest}"


def _ensure_decodable(pdf, xref: int, path: Path, extension: str) -> tuple[Path, str] | None:
    """Re-encode embedded images Pillow/VLM pipelines can't reliably decode.

    PyMuPDF can extract exotic codecs (e.g. JPEG2000/"jpx") straight from the PDF,
    but Pillow's JP2 decoder frequently raises "broken data stream" on real-world
    scans, and most VLM APIs don't accept those mime types either. Re-render such
    images to PNG via the page's own pixmap so downstream consumers always get a
    format they can open. Returns None if no usable image could be produced,
    so the caller can skip this one image instead of aborting the whole run.
    """
    if extension.lower() in _SAFE_EXTENSIONS:
        try:
            from PIL import Image

            with Image.open(path) as probe:
                probe.load()
            return path, extension
        except Exception:
            pass  # fall through and re-render below

    import fitz

    png_path = path.with_suffix(".png")
    try:
        pixmap = fitz.Pixmap(pdf, xref)
        if pixmap.colorspace is None or pixmap.colorspace.n not in (1, 3):
            # fz_save_pixmap_as_png only accepts grayscale/rgb; force a
            # conversion for CMYK, indexed, DeviceN, or any other colorspace.
            pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
        if pixmap.alpha:
            pixmap = fitz.Pixmap(pixmap, 0)  # drop alpha channel before saving
        pixmap.save(png_path)
    except Exception:
        png_path.unlink(missing_ok=True)
        if path.exists():
            path.unlink(missing_ok=True)
        return None

    if png_path != path:
        path.unlink(missing_ok=True)
    return png_path, "png"
