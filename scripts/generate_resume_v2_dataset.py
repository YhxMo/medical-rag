"""Create 80 source-conditioned AI candidates before any system answer experiment."""

from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.application.client import atomic_json, digest
from src.application.images import prepare_image, registered_image
from src.application.service import BOUNDARY, make_clients
from src.config.settings import load_settings
from src.evaluation.run_manifest import freeze_run_manifest
from src.indexer.common import read_evidence
from scripts.prepare_resume_v2 import sha

PROMPT = (
    BOUNDARY
    + """Create four textbook learning evaluation tasks grounded in the supplied page and original image.
Return JSON {"tasks":[{"kind":"text|chart|screenshot|behavior","question":"English question",
"reference_answer":"source-limited answer in English","expected_behavior":"answer|refuse|correct_premise",
"evidence_ids":["provided ID"],"requires_visual":false,"rationale":"why supported / missing"}]}.
One task per kind. Text question must be answerable from text alone. Chart must require inspecting the actual
figure, not merely its caption. Screenshot question assumes the user uploaded this page; do not ask patient diagnosis.
Behavior task: use the requested behavior_type. Missing-information tasks must ask for a specific absent value,
not knowledge that might be elsewhere in the textbook. False-premise tasks must be explicitly refutable by this page.
Paraphrase, avoid quoting the source, and don't insert page numbers or evidence IDs into questions.
No clinical recommendations. Every evidence_id must be supplied. Unreadable visuals must be flagged,
not converted to invented answers. Return exactly four unique kinds. Never claim human review."""
)


def validate_tasks(raw, allowed, behavior):
    rows = raw.get("tasks")
    kinds = {"text", "chart", "screenshot", "behavior"}
    if (
        not isinstance(rows, list)
        or len(rows) != 4
        or {r.get("kind") for r in rows} != kinds
    ):
        raise ValueError("Require all four task kinds")
    for r in rows:
        for key in ("question", "reference_answer", "rationale"):
            minimum = 1 if key == "reference_answer" else 5
            if (
                not isinstance(r.get(key), str)
                or not minimum <= len(r[key].strip()) <= 2000
            ):
                raise ValueError("Invalid task text")
        if type(r.get("requires_visual")) is not bool:
            raise ValueError("Missing visual requirement")
        if r["kind"] == "chart" and not r["requires_visual"]:
            raise ValueError("Chart task does not require image")
        if r["kind"] == "text" and (
            r["requires_visual"]
            or any(i.startswith("visual_") for i in r["evidence_ids"])
        ):
            raise ValueError("Text task must have only text evidence")
        if r["expected_behavior"] != (
            behavior if r["kind"] == "behavior" else "answer"
        ):
            raise ValueError("Unexpected behavior label")
        if (
            not isinstance(r.get("evidence_ids"), list)
            or not r["evidence_ids"]
            or not set(r["evidence_ids"]) <= allowed
        ):
            raise ValueError("Unknown source label")
    return rows


def validate_splits(rows, manifest):
    groups = {r["group_id"]: r for r in manifest["visual_pages"]}
    by_hash = {}
    for q in rows:
        group = groups[q["group_id"]]
        if q["split"] != group["split"]:
            raise ValueError("Task split differs from source group")
        for p in group["image_paths"]:
            h = manifest["image_registry"][p]
            if h in by_hash and by_hash[h] != q["split"]:
                raise ValueError("Image leakage across splits")
            by_hash[h] = q["split"]
    for a in rows:
        for b in rows:
            if (
                a["split"] != b["split"]
                and a["source_file"] == b["source_file"]
                and abs(a["page"] - b["page"]) <= 2
            ):
                raise ValueError("Adjacent-page leakage")


def generate(settings):
    base = ROOT / "artifacts/resume-v2"
    out = ROOT / "data/evaluation/resume-v2"
    manifest = json.loads((base / "preparation.json").read_text())
    evidence = read_evidence(base / "text_baseline.jsonl")
    _, client = make_clients(settings)
    if not client.available:
        raise RuntimeError(
            "Visual API credential required for actual multimodal task generation"
        )
    freeze_run_manifest(
        out,
        {
            "preparation": sha(base / "preparation.json"),
            "prompt": digest(PROMPT),
            "model": client.config["model"],
            "endpoint": client.config["api_base"],
        },
    )
    if (out / "dataset.json").exists():
        rows = json.loads((out / "dataset.json").read_text())
        validate_splits(rows, manifest)
        return rows
    selected = []
    for split in ("dev", "holdout"):
        pool = [r for r in manifest["visual_pages"] if r["split"] == split]
        if len(pool) < 10:
            raise ValueError("Insufficient independent groups")
        selected.extend(pool[int((i + 0.5) * len(pool) / 10)] for i in range(10))
    atomic_json(out / "sampling_plan.json", selected)
    tasks = []
    for i, row in enumerate(selected):
        chunks = [
            e
            for e in evidence
            if e.source_file == row["source_file"] and e.page_start == row["page"]
        ]
        refs = [{"id": e.evidence_id, "content": e.content} for e in chunks]
        visual_id = "visual_" + digest(row["group_id"])[:20]
        refs.append(
            {
                "id": visual_id,
                "content": "Original image for this page is supplied separately.",
            }
        )
        behavior = "refuse" if i % 2 == 0 else "correct_premise"
        path = out / f"batch_{i:02d}.json"
        if path.exists():
            raw = json.loads(path.read_text())
        else:
            raw, _ = client.call(
                PROMPT,
                {"sources": refs, "behavior_type": behavior},
                images=[
                    prepare_image(
                        registered_image(
                            row["image_paths"][0], manifest["image_registry"]
                        )
                    )[0]
                ],
                version=sha(base / "preparation.json"),
                category="evaluation",
                max_tokens=2048,
            )
            atomic_json(path, raw)
        for repair in range(3):
            try:
                batch = validate_tasks(raw, {r["id"] for r in refs}, behavior)
                break
            except ValueError as exc:
                if repair == 2:
                    raise
                repaired_path = out / f"batch_{i:02d}_repair_{repair + 1}.json"
                if repaired_path.exists():
                    raw = json.loads(repaired_path.read_text())
                else:
                    raw, _ = client.call(
                        PROMPT,
                        {
                            "sources": refs,
                            "behavior_type": behavior,
                            "previous_draft": raw,
                            "validation_error": str(exc),
                            "revision_instruction": "Revise the task content using the original image and sources. Do not merely flip requires_visual to pass validation. A chart task must require a visible layout, shape, axis, or relationship absent from the extracted text. If impossible, keep requires_visual false and explain why. Preserve other valid tasks.",
                        },
                        images=[
                            prepare_image(
                                registered_image(
                                    row["image_paths"][0], manifest["image_registry"]
                                )
                            )[0]
                        ],
                        version=sha(base / "preparation.json"),
                        category="evaluation",
                        max_tokens=2048,
                    )
                    atomic_json(repaired_path, raw)
        for r in batch:
            q = {
                "question_id": f"resume_v2_{i:02d}_{r['kind']}",
                **r,
                "expected_evidence_ids": r["evidence_ids"],
                "expected_answer": r["reference_answer"],
                "group_id": row["group_id"],
                "split": row["split"],
                "source_file": row["source_file"],
                "page": row["page"],
                "image_path": row["image_paths"][0]
                if r["kind"] == "screenshot"
                else None,
                "gold_image": row["image_paths"][0],
                "reviewed": False,
                "AI_generated": True,
            }
            tasks.append(q)
        print(f"Task group {i + 1}/20 generated", flush=True)
    validate_splits(tasks, manifest)
    corrections_path = out / "draft_source_review.json"
    if corrections_path.exists():
        corrections = json.loads(corrections_path.read_text())
        by_id = {q["question_id"]: q for q in tasks}
        for change in corrections:
            q = by_id[change["question_id"]]
            if q["reference_answer"] != change["original_reference_answer"]:
                raise ValueError("Source correction no longer matches draft")
            q["reference_answer"] = q["expected_answer"] = change[
                "corrected_reference_answer"
            ]
            q["AI_source_correction"] = change

    # Separate source audit is saved, and failures stay visible rather than being dropped.
    def audit_task(q):
        path = out / f"{q['question_id']}_audit.json"
        if path.exists():
            audit = json.loads(path.read_text())
        else:
            sources = [
                {"id": e.evidence_id, "content": e.content}
                for e in evidence
                if e.evidence_id in q["expected_evidence_ids"]
            ]
            audit, _ = client.call(
                BOUNDARY
                + "Audit this draft task against original sources. Return JSON "
                '{"supported":true,"reason":"..."}. For refusal tasks check that the requested value is '
                "absent from these sources; for false premises check the correction. Flag uncertainty.",
                {"task": q, "sources": sources},
                images=[prepare_image(q["gold_image"])[0]],
                category="evaluation",
                version=sha(base / "preparation.json"),
                max_tokens=512,
            )
            if type(audit.get("supported")) is not bool or not isinstance(
                audit.get("reason"), str
            ):
                raise ValueError("Invalid audit")
            atomic_json(path, audit)
        q["AI_source_audit"] = audit
        return q

    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = list(pool.map(audit_task, tasks))
    atomic_json(out / "dataset.json", tasks)
    atomic_json(
        out / "frozen.json",
        {
            "count": len(tasks),
            "dataset_sha256": sha(out / "dataset.json"),
            "AI_audit_flagged": sum(
                not q["AI_source_audit"]["supported"] for q in tasks
            ),
            "human_reviewed": 0,
            "AI_source_corrected": sum("AI_source_correction" in q for q in tasks),
            "source_corrections_sha256": sha(corrections_path)
            if corrections_path.exists()
            else None,
            "holdout_status": "not_yet_run",
        },
    )
    return tasks


if __name__ == "__main__":
    generate(load_settings(ROOT / "config.resume-v2.yaml"))
