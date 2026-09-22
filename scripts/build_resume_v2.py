"""Build immutable text or caption-enhanced indexes; optional three-page VLM pilot."""

from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.application.client import atomic_json, digest
from src.application.images import (
    prepare_image,
    registered_image,
    normalize_visual_fields,
)
from src.application.service import BOUNDARY, make_clients
from src.config.settings import load_settings
from src.indexer.common import read_evidence, retrieval_text, evidence_from_json
from src.indexer.qdrant_store import QdrantIndexStore
from src.schema import EvidenceItem
from src.evaluation.run_manifest import freeze_run_manifest
from scripts.prepare_resume_v2 import prepare, sha

CAPTION = BOUNDARY + (
    "Describe the supplied textbook page in English for retrieval. Return JSON with "
    "visible_text, visual_relations, units, uncertainties, summary (all strings). "
    "Do not infer curve values, diagnoses or missing formulas. Separate visible evidence from interpretation. "
    "Nearby text is only context, not a license to invent visual content."
)


def build(mode, settings):
    manifest = prepare()
    root = ROOT / "artifacts/resume-v2"
    variants = {
        name: read_evidence(root / f"text_{name}.jsonl")
        for name in ("baseline", "chapter")
    }
    captions = []
    if mode == "multimodal":
        _, vision = make_clients(settings)
        if not vision.available:
            raise RuntimeError("DASHSCOPE_API_KEY is required for real visual indexing")
        records = []
        # First one page from each source, followed by the remainder.
        pages = manifest["visual_pages"]
        pilot = [
            next(r for r in pages if r["source_file"] == s)
            for s in sorted({r["source_file"] for r in pages})
        ]
        ordered = pilot + [r for r in pages if r not in pilot]
        limits_path = root / "caption-run/output_limits.json"
        limits = json.loads(limits_path.read_text()) if limits_path.exists() else {}
        freeze_run_manifest(
            root / "caption-run",
            {
                "preparation": sha(root / "preparation.json"),
                "model": vision.config["model"],
                "endpoint": vision.config["api_base"],
                "prompt": digest(CAPTION),
            },
        )
        for i, row in enumerate(ordered):
            path = registered_image(row["image_paths"][0], manifest["image_registry"])
            text = "\n".join(
                e.content
                for e in variants["baseline"]
                if e.source_file == row["source_file"] and e.page_start == row["page"]
            )[:6000]
            output, usage = vision.call(
                CAPTION,
                {"nearby_text": text},
                images=[prepare_image(path)[0]],
                category="ingest",
                version=sha(root / "preparation.json"),
                max_tokens=limits.get(row["group_id"], 2048),
            )
            output, conversions = normalize_visual_fields(
                output,
                (
                    "visible_text",
                    "visual_relations",
                    "units",
                    "uncertainties",
                    "summary",
                ),
            )
            records.append(
                {
                    "page": row,
                    "description": output,
                    "usage": usage,
                    "normalized_text_fields": conversions,
                }
            )
            atomic_json(root / "caption-run/descriptions.json", records)
            if i == 2:
                atomic_json(
                    root / "caption-run/pilot.json",
                    {
                        "pages": 3,
                        "status": "API_and_schema_passed_not_expert_validation",
                        "requested_model": vision.config["model"],
                        "returned_models": [r["usage"]["model"] for r in records],
                    },
                )
            print(f"Visual page {i + 1}/{len(ordered)} complete", flush=True)
        for row in records:
            page, desc = row["page"], row["description"]
            captions.append(
                EvidenceItem(
                    "visual_" + digest(page["group_id"])[:20],
                    "image_caption",
                    page["source_file"],
                    "\n".join(f"{k}: {v}" for k, v in desc.items()),
                    page["page"],
                    page["page"],
                    {
                        "derived_by_model": True,
                        "image_paths": page["image_paths"][:1],
                        **page["chapter"],
                    },
                )
            )
    output = root / mode
    input_manifest = {
        "mode": mode,
        "preparation": sha(root / "preparation.json"),
        "captions": digest([e.content for e in captions]),
        "embedding": settings.get("embedding", "fastembed", "model"),
        "text_count": len(variants["baseline"]),
        "caption_count": len(captions),
    }
    if (output / "build_manifest.json").exists():
        if json.loads((output / "build_manifest.json").read_text()) != input_manifest:
            raise ValueError(
                "Refusing to overwrite a different completed index version"
            )
        print("Matching completed indexes already exist")
        return
    from src.cli import _build_embedding_provider

    provider = _build_embedding_provider("fastembed", settings)
    for name, evidence in variants.items():
        rows = evidence + [
            replace(
                e,
                metadata={
                    **e.metadata,
                    "retrieval_heading": " > ".join(e.metadata["chapter_path"])
                    if name == "chapter"
                    else "",
                },
            )
            for e in captions
        ]
        store = QdrantIndexStore(
            output / name / "index",
            collection_name=f"resume_v2_{name}",
            local_path=output / name / "qdrant",
        )
        effective_provider = provider
        if mode == "multimodal":
            previous_manifest = root / "text/build_manifest.json"
            if (
                previous_manifest.exists()
                and json.loads(previous_manifest.read_text())["embedding"]
                == input_manifest["embedding"]
            ):
                from src.embedding.cached import CachedEmbeddingProvider, text_key

                previous = QdrantIndexStore(
                    root / "text" / name / "index",
                    collection_name=f"resume_v2_{name}",
                    local_path=root / "text" / name / "qdrant",
                )
                vectors = {}
                try:
                    client, _ = previous._connection()
                    offset = None
                    while True:
                        points, offset = client.scroll(
                            collection_name=previous.collection_name,
                            offset=offset,
                            limit=256,
                            with_payload=True,
                            with_vectors=True,
                        )
                        for point in points:
                            item = evidence_from_json(point.payload)
                            vectors[text_key(retrieval_text(item))] = point.vector
                        if offset is None:
                            break
                finally:
                    if previous._client is not None:
                        previous._client.close()
                effective_provider = CachedEmbeddingProvider(provider, vectors)
        try:
            store.build(rows, effective_provider)
        finally:
            if store._client is not None:
                store._client.close()
        print(f"{mode}/{name}: {len(rows)} records", flush=True)
    atomic_json(output / "build_manifest.json", input_manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "multimodal"], default="text")
    args = parser.parse_args()
    build(args.mode, load_settings(ROOT / "config.resume-v2.yaml"))
