"""Publish verification facts without inventing unfinished online results."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.application.client import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    out = ROOT / "docs/resume-v2"
    local = ROOT / "artifacts/resume-v2"
    status = {
        "scope": "textbook learning application; not clinical validation",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "human_reviewed": 0,
        "additional_paid_api_calls": 0,
        "live_multimodal_accepted": False,
        "new_80_task_evaluation_completed": False,
    }
    prep = local / "preparation.json"
    if prep.exists():
        data = json.loads(prep.read_text())
        status.update(
            text_records=data["text_count"],
            selected_visual_pages=len(data["visual_pages"]),
            registered_image_files=len(data["image_registry"]),
            source_split_counts=data["split_counts"],
            chapter_counts=data["chapter_counts"],
        )
    status["text_indexes_built"] = (local / "text/build_manifest.json").exists()
    status["multimodal_indexes_built"] = (
        local / "multimodal/build_manifest.json"
    ).exists()
    ledger = local / "budget.json"
    if ledger.exists():
        data = json.loads(ledger.read_text())
        status["additional_paid_api_calls"] = len(data["calls"])
        status["accounted_cost_upper_cny"] = sum(
            r["charged_upper_cny"] for r in data["calls"]
        )
    else:
        status["accounted_cost_upper_cny"] = 0
    if args.run_tests:
        test = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--tb=short",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        (local / "test-output.txt").write_text(test.stdout + test.stderr)
        count = re.search(r"(\d+) passed", test.stdout)
        status["tests"] = {
            "exit_code": test.returncode,
            "passed": int(count[1]) if count else 0,
            "output_sha256": hashlib.sha256(
                (test.stdout + test.stderr).encode()
            ).hexdigest(),
        }
        if test.returncode:
            atomic_json(out / "status.json", status)
            raise SystemExit(test.returncode)
    else:
        prior = out / "status.json"
        if prior.exists():
            status["tests"] = json.loads(prior.read_text()).get("tests")
    regression = local / "regression-retrieval/summary.json"
    for source, target in (
        (local / "dev-retrieval/summary.json", "retrieval_development.json"),
        (local / "cold_process_benchmark.json", "performance_cold.json"),
    ):
        if source.exists():
            atomic_json(out / target, json.loads(source.read_text()))
    if regression.exists():
        report = json.loads(regression.read_text())
        atomic_json(out / "retrieval_regression.json", report)
        status["regression_strategies_completed"] = [
            name for name, value in report["reports"].items() if value["failed"] == 0
        ]
    data_root = ROOT / "data/evaluation/resume-v2"
    frozen = data_root / "frozen.json"
    status["new_task_count"] = (
        json.loads(frozen.read_text())["count"] if frozen.exists() else 0
    )
    status["pending"] = []
    if not status["multimodal_indexes_built"]:
        status["pending"].append("Visual API pilot and 60 real image descriptions")
    if not frozen.exists():
        status["pending"].append("80 real source-audited AI candidate tasks")
    all_answers_complete = status["new_task_count"] == 80
    all_reviews_complete = True
    for split in ("dev", "holdout"):
        report = data_root / f"answers-{split}/summary.json"
        review = data_root / f"answers-{split}/review_completed.json"
        if not report.exists():
            all_answers_complete = False
            status["pending"].append(
                f"{split} end-to-end answer/vision/refusal experiments"
            )
        else:
            result = json.loads(report.read_text())
            atomic_json(out / f"answers_{split}.json", result)
            repair_path = report.parent / "judge-repair/summary.json"
            graded = result["judged"]
            if repair_path.exists():
                repair = json.loads(repair_path.read_text())
                atomic_json(out / f"judge_repair_{split}.json", repair)
                graded = repair["effective_judged"]
            all_answers_complete &= (
                graded == result["expected_runs"]
                and result["completed_answers"] == result["expected_runs"]
            )
        if not review.exists():
            all_reviews_complete = False
            status["pending"].append(
                f"{split} source review of failures, disagreements and random sample"
            )
        else:
            atomic_json(
                out / f"source_review_{split}.json", json.loads(review.read_text())
            )
    status["new_80_task_evaluation_completed"] = all_answers_complete
    status["live_multimodal_accepted"] = (
        status["multimodal_indexes_built"]
        and all_answers_complete
        and all_reviews_complete
    )
    status["pending"].extend(
        [
            "Remote Linux CI has not been observed",
            "Clinical or medical expert validation is outside scope",
        ]
    )
    status["required_credentials"] = ["DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"]
    fingerprint = hashlib.sha256()
    for directory in ("src", "scripts", "tests"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            if path.name.startswith("._"):
                continue
            fingerprint.update(str(path.relative_to(ROOT)).encode())
            fingerprint.update(path.read_bytes())
    status["python_sources_sha256"] = fingerprint.hexdigest()
    atomic_json(out / "status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
