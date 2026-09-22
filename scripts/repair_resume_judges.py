"""One explicit judge-only repair pass; original answers and grades are immutable."""

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
from src.application.service import make_clients
from src.config.settings import load_settings
from src.evaluation.run_manifest import freeze_run_manifest
from src.evaluation.resume_reporting import operational_metrics
from src.indexer.common import read_evidence
from scripts.evaluate_resume_v2 import JUDGE, judge_valid
from scripts.prepare_resume_v2 import sha

PROMPT = (
    JUDGE
    + """ This is one repair of an invalid grader response, NOT a new answer generation.
Numbered retrieved sources explicitly map citation_number to the answer's [n] references.
Return exactly the required_citation_ids, once each, including [] when empty. Do not invent missing assessments.
Gold sources and the first gold image do not count as evidence available to the answerer unless present in retrieved evidence.
Scientific plausibility or familiar values alone cannot establish grounding. Verify each numerical claim explicitly.
Keep the reason under 120 English words. Scores and flags must retain the specified JSON types."""
)


def repair(split):
    root = ROOT / "data/evaluation/resume-v2"
    out = root / f"answers-{split}"
    tasks = json.loads((root / "dataset.json").read_text())
    questions = {q["question_id"]: q for q in tasks}
    records = [
        json.loads(p.read_text())
        for p in sorted(out.glob("resume_v2_*.json"))
        if not p.name.endswith("_judge_raw.json")
    ]
    failed = [
        r
        for r in records
        if r["status"] != "ok" and r["result"]["status"] != "service_error"
    ]
    target = out / "judge-repair"
    _, client = make_clients(load_settings(ROOT / "config.resume-v2.yaml"))
    freeze_run_manifest(
        target,
        {
            "dataset": sha(root / "dataset.json"),
            "prompt": digest(PROMPT),
            "model": client.config["model"],
            "initial_summary": sha(out / "summary.json"),
        },
    )
    evidence = {
        e.evidence_id: e
        for e in read_evidence(ROOT / "artifacts/resume-v2/text_baseline.jsonl")
    }

    def one(row):
        key = f"{row['question_id']}_{row['mode']}"
        path = target / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        q, answer = questions[row["question_id"]], row["result"]
        refs = answer.get("citations", {}).get("unique_citations", [])
        original = out / f"{key}_judge_raw.json"
        result = {
            "question_id": row["question_id"],
            "mode": row["mode"],
            "status": "failed",
            "original_untouched": True,
        }
        try:
            grade, usage = client.call(
                PROMPT,
                {
                    "question": q["question"],
                    "reference": q["reference_answer"],
                    "answer": answer["answer"],
                    "expected_behavior": q["expected_behavior"],
                    "required_citation_ids": refs,
                    "initial_invalid_grade": json.loads(original.read_text())
                    if original.exists()
                    else None,
                    "retrieved_sources": [
                        {"citation_number": i, **s}
                        for i, s in enumerate(answer["sources"], 1)
                    ],
                    "gold_sources": [
                        {"id": k, "content": evidence[k].content}
                        for k in q["expected_evidence_ids"]
                        if k in evidence
                    ],
                    "answer_received_originals": bool(
                        answer["image_paths"] or q.get("image_path")
                    ),
                    "image_order": ["gold_original"]
                    + ["answerer_retrieved_original"] * len(answer["image_paths"][:2]),
                },
                images=[prepare_image(q["gold_image"])[0]]
                + [prepare_image(p)[0] for p in answer["image_paths"][:2]],
                category="evaluation",
                version=sha(root / "dataset.json"),
                max_tokens=1536,
            )
            atomic_json(target / f"{key}.raw.json", grade)
            if isinstance(grade.get("citations"), list):
                grade = {
                    **grade,
                    "citations": [c for c in grade["citations"] if c.get("id") in refs],
                }
            result.update(status="ok", judge=judge_valid(grade, refs), usage=usage)
        except Exception as exc:
            result["error"] = type(exc).__name__
        atomic_json(path, result)
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        repaired = list(pool.map(one, failed))
    replacements = {
        (r["question_id"], r["mode"]): r for r in repaired if r["status"] == "ok"
    }
    effective = [
        {
            **r,
            "judge": replacements[(r["question_id"], r["mode"])]["judge"],
            "status": "ok",
        }
        if (r["question_id"], r["mode"]) in replacements
        else r
        for r in records
    ]
    summary = {
        "initial_invalid_grades": len(failed),
        "repaired_grades": len(replacements),
        "repair_failures": len(failed) - len(replacements),
        "total_answer_runs": len(records),
        "effective_judged": sum(r["status"] == "ok" for r in effective),
        "original_answers_and_grades_unchanged": True,
        "protocol": "One judge-only repair pass, predeclared before holdout",
        "operations_with_repaired_grades": operational_metrics(effective, tasks),
    }
    atomic_json(target / "summary.json", summary)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "operations_with_repaired_grades"}
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "holdout"], required=True)
    repair(p.parse_args().split)
