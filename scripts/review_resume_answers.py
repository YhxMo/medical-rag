"""Second AI source review; preserve initial answers and grades, never label human review."""

from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.application.client import atomic_json, digest
from src.application.images import prepare_image
from src.application.service import BOUNDARY, make_clients
from src.config.settings import load_settings
from src.evaluation.run_manifest import freeze_run_manifest
from scripts.prepare_resume_v2 import sha

REVIEW = (
    BOUNDARY
    + """Review a saved answer and its initial AI grade against numbered retrieved evidence.
Use ONLY the numbered retrieved passages and images actually supplied to the answerer to assess grounding.
The first image is a GOLD reference: if answer_received_originals=false it was NOT available to the answerer.
Never penalize grounding for absence of information that occurs only in gold evidence.
Gold may assess answer correctness and completeness, but does NOT make an unsupported answer claim grounded.
Familiar textbook knowledge, typical values and statements consistent with science are NOT evidence support.
Check each numerical value, unit and directional relationship against an explicit passage or visible original.
Citation numbers refer to citation_number, not the opaque source ID. Review exactly required_citation_ids.
Do not credit the initial grade merely because it sounds plausible. Report contradictions in that grade.
Return JSON {"original_grade_reliable":true,"material_unsupported_claims":["brief claim and missing evidence"],
"citation_review":[{"id":1,"supported":true}],"visual_understanding":"brief source-limited finding",
"reason":"brief explanation"}. Use empty arrays when appropriate. This is AI review, not expert validation.
Keep all prose concise, total under 250 English words."""
)


def review(split):
    root = ROOT / "data/evaluation/resume-v2"
    out = root / f"answers-{split}"
    queue = json.loads((out / "review_queue.json").read_text())
    tasks = {
        q["question_id"]: q for q in json.loads((root / "dataset.json").read_text())
    }
    records = [
        json.loads(p.read_text())
        for p in sorted(out.glob("resume_v2_*.json"))
        if not p.name.endswith("_judge_raw.json")
    ]
    selected = [r for r in records if r["question_id"] in queue["ids"]]
    review_dir = out / "source-review"
    _, client = make_clients(load_settings(ROOT / "config.resume-v2.yaml"))
    freeze_run_manifest(
        review_dir,
        {
            "dataset": sha(root / "dataset.json"),
            "queue": sha(out / "review_queue.json"),
            "prompt": digest(REVIEW),
            "model": client.config["model"],
        },
    )

    def one(row):
        path = review_dir / f"{row['question_id']}_{row['mode']}.json"
        if path.exists():
            return json.loads(path.read_text())
        q, result = tasks[row["question_id"]], row["result"]
        refs = result.get("citations", {}).get("unique_citations", [])
        payload = {
            "question": q["question"],
            "reference_answer": q["reference_answer"],
            "expected_behavior": q["expected_behavior"],
            "answer": result["answer"],
            "answer_status": result["status"],
            "initial_grade": row.get("judge"),
            "initial_judge_error": row.get("judge_error"),
            "required_citation_ids": refs,
            "numbered_retrieved_sources": [
                {"citation_number": i, **s} for i, s in enumerate(result["sources"], 1)
            ],
            "answer_received_originals": bool(
                result["image_paths"] or q.get("image_path")
            ),
            "image_order": ["gold_reference"]
            + ["answerer_retrieved_original"] * len(result["image_paths"][:2]),
        }
        value, usage = client.call(
            REVIEW,
            payload,
            images=[prepare_image(q["gold_image"])[0]]
            + [prepare_image(p)[0] for p in result["image_paths"][:2]],
            category="evaluation",
            version=sha(root / "dataset.json"),
            max_tokens=1536,
        )
        atomic_json(path.with_suffix(".raw.json"), value)
        if (
            type(value.get("original_grade_reliable")) is not bool
            or not isinstance(value.get("material_unsupported_claims"), list)
            or not all(isinstance(v, str) for v in value["material_unsupported_claims"])
            or not all(
                isinstance(value.get(k), str)
                for k in ("visual_understanding", "reason")
            )
        ):
            raise ValueError("Invalid source review schema")
        citations = value.get("citation_review", [])
        if isinstance(citations, list):
            relevant = [c for c in citations if c.get("id") in refs]
            value["uncited_entries_ignored"] = [
                c for c in citations if c.get("id") not in refs
            ]
            citations = value["citation_review"] = relevant
        if (
            not isinstance(citations, list)
            or {c.get("id") for c in citations} != set(refs)
            or len(citations) != len(refs)
            or any(type(c.get("supported")) is not bool for c in citations)
        ):
            raise ValueError("Incomplete source review citations")
        record = {
            "question_id": row["question_id"],
            "mode": row["mode"],
            "review": value,
            "usage": usage,
            "AI_reviewed": True,
            "human_reviewed": False,
            "original_answer_preserved": True,
        }
        atomic_json(path, record)
        return record

    with ThreadPoolExecutor(max_workers=2) as pool:
        reviews = list(pool.map(one, selected))
    summary = {
        "scope": "Second AI source review, same provider; not independent medical expertise",
        "question_ids": queue["ids"],
        "answers_reviewed": len(reviews),
        "human_reviewed": 0,
        "initial_grades_disputed": sum(
            not r["review"]["original_grade_reliable"] for r in reviews
        ),
        "answers_with_unsupported_claims": sum(
            bool(r["review"]["material_unsupported_claims"]) for r in reviews
        ),
        "prompt_sha256": digest(REVIEW),
        "original_answers_and_grades_unchanged": True,
    }
    atomic_json(out / "review_completed.json", summary)
    if split == "dev":
        atomic_json(
            ROOT / "artifacts/resume-v2/dev_review_decision.json",
            {
                "decision": "freeze_system_and_report_observed_limitations",
                "review_sha256": sha(out / "review_completed.json"),
                "dataset_sha256": sha(root / "dataset.json"),
                "limitations": "Initial grades can be unreliable; publish review disagreements and no clinical accuracy claims.",
                "answer_system_changed_after_dev": False,
            },
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "holdout"], required=True)
    review(p.parse_args().split)
