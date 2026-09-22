"""Prepare isolated, source-verified text records and 60 frozen visual pages."""

from __future__ import annotations
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.application.client import atomic_json, digest
from src.indexer.common import read_evidence, write_evidence


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chapter_map(doc):
    """Bookmarks are authoritative; conservative numbered headings are fallback."""
    bookmarks = {}
    stack = []
    for level, title, page in doc.get_toc():
        if not 1 <= page <= len(doc):
            continue
        stack = stack[: level - 1]
        stack.append(title.strip())
        bookmarks[page] = list(stack)
    current, output = [], {}
    for number, page in enumerate(doc, 1):
        origin = "unknown"
        if bookmarks:
            if number in bookmarks:
                current = bookmarks[number]
            origin = "bookmark" if current else "unknown"
        else:
            # Require number + substantial heading, exclude numerical table cells.
            for line in page.get_text().splitlines()[:30]:
                clean = line.strip()
                if re.match(
                    r"^\d{1,2}(?:\.\d{1,2}){1,3}\.?\s+[A-Z][A-Z ,:/()\-]{7,100}$", clean
                ):
                    current = [clean]
                    break
            origin = "numbered_heading" if current else "unknown"
        output[number] = {"chapter_path": list(current), "chapter_origin": origin}
    return output


def kind(text):
    low = text.lower()
    if re.search(r"curve|graph|plot|relationship", low):
        return "curve"
    if re.search(r"\btable\s+\d", low):
        return "table"
    if re.search(r"ultrasound|sonogram|radiograph|ct image", low):
        return "imaging_example"
    return "diagram"


def prepare(root=ROOT):
    import pymupdf as fitz

    root = Path(root)
    out = root / "artifacts/resume-v2"
    manifest_path = out / "preparation.json"
    sources = json.loads((root / "docs/evaluation/public_corpus_v1.json").read_text())
    original = root / "artifacts/public-v1/index/evidence.jsonl"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["original_evidence_sha256"] != sha(original):
            raise ValueError("Source index changed; prepare a new version")
        for row in sources["documents"]:
            if sha(root / row["file"]) != row["sha256"]:
                raise ValueError("Source PDF changed")
        for path, expected in saved["image_registry"].items():
            if sha(path) != expected:
                raise ValueError("Prepared image changed")
        print(
            json.dumps(
                {
                    "status": "reused_frozen_preparation",
                    "pages": len(saved["visual_pages"]),
                }
            )
        )
        return saved
    evidence = read_evidence(original)
    enhanced, visual_pages, registry, by_source_page = [], [], {}, {}
    image_root = out / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    old_questions = (
        root / "data/evaluation/medical-review-v3/regression_candidates.json"
    )
    old_ids = set()
    if old_questions.exists():
        for q in json.loads(old_questions.read_text()):
            old_ids.update(q["expected_evidence_ids"])
    excluded = {
        (e.source_file, p)
        for e in evidence
        if e.evidence_id in old_ids
        for p in range(max(1, e.page_start - 2), e.page_start + 3)
    }
    for source in sources["documents"]:
        pdf = root / source["file"]
        if sha(pdf) != source["sha256"]:
            raise ValueError("Source PDF checksum mismatch")
        doc = fitz.open(pdf)
        chapters = chapter_map(doc)
        candidates = []
        for number, page in enumerate(doc, 1):
            text = page.get_text()
            if number < 16 or number > len(doc) * 0.9 or len(text.strip()) < 160:
                continue
            if (pdf.name, number) in excluded:
                continue
            if not re.search(r"\b(fig(?:ure)?\.?|table)\s*\d", text, re.I):
                continue
            substantial_images = any(
                info[2] >= 128 and info[3] >= 128 for info in page.get_images(full=True)
            )
            caption_line = bool(
                re.search(
                    r"^\s*(?:FIG\.?|Fig\.?|Figure|FIGURE|TABLE|Table)\s+\d", text, re.M
                )
            )
            if not substantial_images and not (
                caption_line and len(page.get_drawings()) >= 5
            ):
                continue
            candidates.append({"page": number, "kind": kind(text), "text": text})
        # Stratify by kind then spatial coverage, maintain >2-page separation.
        selected = []
        groups = {
            k: [r for r in candidates if r["kind"] == k]
            for k in ("curve", "table", "imaging_example", "diagram")
        }
        for rows in groups.values():
            if rows:
                for i in range(min(5, len(rows))):
                    r = rows[int((i + 0.5) * len(rows) / min(5, len(rows)))]
                    if all(abs(r["page"] - s["page"]) > 4 for s in selected):
                        selected.append(r)
        for r in sorted(
            candidates,
            key=lambda r: hashlib.sha256(
                f"{pdf.name}:{r['page']}".encode()
            ).hexdigest(),
        ):
            if len(selected) >= 20:
                break
            if all(abs(r["page"] - s["page"]) > 4 for s in selected):
                selected.append(r)
        if len(selected) < 20:
            raise ValueError(
                f"Only {len(selected)} independent visual pages in {pdf.name}; do not silently reduce sample"
            )
        for i, item in enumerate(sorted(selected[:20], key=lambda r: r["page"])):
            number = item["page"]
            page = doc[number - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
            raw = pix.tobytes("png")
            h = hashlib.sha256(raw).hexdigest()
            path = image_root / f"{h}.png"
            if not path.exists():
                path.write_bytes(raw)
            paths = [str(path.resolve())]
            registry[paths[0]] = h
            # Extract only reasonably sized embedded images; exact hashes deduplicate globally.
            for info in page.get_images(full=True):
                extracted = doc.extract_image(info[0])
                if min(extracted["width"], extracted["height"]) < 128:
                    continue
                try:
                    pix = fitz.Pixmap(doc, info[0])
                    if pix.colorspace is None:
                        continue
                    if pix.colorspace.n not in (1, 3):
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    if pix.alpha:
                        pix = fitz.Pixmap(pix, 0)
                    raw = pix.tobytes("png")
                    h = hashlib.sha256(raw).hexdigest()
                    ep = image_root / f"{h}.png"
                    if not ep.exists():
                        ep.write_bytes(raw)
                    registry[str(ep.resolve())] = h
                    if str(ep.resolve()) not in paths:
                        paths.append(str(ep.resolve()))
                except Exception:
                    continue
            group = f"{pdf.stem}:p{number}"
            row = {
                "group_id": group,
                "source_file": pdf.name,
                "page": number,
                "split": "dev" if i % 2 == 0 else "holdout",
                "kind": item["kind"],
                "image_paths": paths,
                "chapter": chapters[number],
                "selection_reason": "Actual raster >=128px or caption plus >=5 drawing paths; >160 text chars; stratified type and page spacing; old labels +/-2 excluded",
            }
            visual_pages.append(row)
            by_source_page[(pdf.name, number)] = row
        for e in evidence:
            if e.source_file != pdf.name:
                continue
            chapter = chapters[e.page_start]
            enhanced.append(
                replace(
                    e,
                    metadata={
                        **e.metadata,
                        **chapter,
                        "source_sha256": source["sha256"],
                        "retrieval_heading": " > ".join(chapter["chapter_path"]),
                        "image_paths": by_source_page.get(
                            (e.source_file, e.page_start), {}
                        ).get("image_paths", [])[:1],
                    },
                )
            )
        doc.close()
    # Same embedded image must not leak between dataset splits. Merge connected groups.
    parent = list(range(len(visual_pages)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owners = {}
    for i, row in enumerate(visual_pages):
        for p in row["image_paths"]:
            h = registry[p]
            if h in owners:
                parent[find(i)] = find(owners[h])
            else:
                owners[h] = i
    for i, row in enumerate(visual_pages):
        row["split"] = visual_pages[find(i)]["split"]
        row["leakage_group"] = visual_pages[find(i)]["group_id"]
    write_evidence(
        out / "text_baseline.jsonl",
        [
            replace(
                e,
                metadata={
                    **e.metadata,
                    "image_paths": by_source_page.get(
                        (e.source_file, e.page_start), {}
                    ).get("image_paths", [])[:1],
                },
            )
            for e in evidence
        ],
    )
    write_evidence(out / "text_chapter.jsonl", enhanced)
    manifest = {
        "version": "resume-v2",
        "original_evidence_sha256": sha(original),
        "source_sha256": {
            Path(s["file"]).name: s["sha256"] for s in sources["documents"]
        },
        "text_count": len(evidence),
        "visual_pages": visual_pages,
        "image_registry": registry,
        "chapter_counts": dict(Counter(e.metadata["chapter_origin"] for e in enhanced)),
        "split_counts": dict(Counter(r["split"] for r in visual_pages)),
        "selection_frozen_before_model_calls": True,
        "human_reviewed": 0,
        "preparation_input_digest": digest(
            {"sources": sources, "evidence": sha(original)}
        ),
    }
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                k: v
                for k, v in manifest.items()
                if k not in {"visual_pages", "image_registry", "source_sha256"}
            },
            indent=2,
        )
    )
    return manifest


if __name__ == "__main__":
    prepare()
