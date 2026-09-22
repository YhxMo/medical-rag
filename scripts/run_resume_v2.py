"""Resume complete workflow; stops explicitly when paid prerequisites are absent."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.application.client import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()
    os.chdir(ROOT)
    commands = [
        ["scripts/prepare_resume_v2.py"],
        ["scripts/build_resume_v2.py", "--mode", "text"],
        ["-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short"],
    ]
    if not args.offline:
        # Check before expensive preparation; no keys are printed or stored.
        missing = [
            k
            for k in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY")
            if not os.environ.get(k)
        ]
        if missing:
            print(
                "Missing runtime credentials: "
                + ", ".join(missing)
                + ". Use scripts/resume_v2_secure_run.py."
            )
            return 2
        commands.extend(
            [
                ["scripts/build_resume_v2.py", "--mode", "multimodal"],
                ["scripts/generate_resume_v2_dataset.py"],
                ["scripts/evaluate_resume_v2.py", "retrieval"],
                ["scripts/evaluate_resume_v2.py", "answers", "--split", "dev"],
                ["scripts/repair_resume_judges.py", "--split", "dev"],
                ["scripts/review_resume_answers.py", "--split", "dev"],
                ["scripts/evaluate_resume_v2.py", "answers", "--split", "holdout"],
                ["scripts/repair_resume_judges.py", "--split", "holdout"],
                ["scripts/review_resume_answers.py", "--split", "holdout"],
            ]
        )
    for command in commands:
        code = subprocess.call([sys.executable, *command], cwd=ROOT)
        if code:
            print(
                "Workflow stopped; previous results preserved. Failed step: "
                + command[0]
            )
            return code
    atomic_json(
        ROOT / "artifacts/resume-v2/workflow_status.json",
        {
            "offline_only": args.offline,
            "completed_commands": commands,
            "human_reviewed": 0,
            "source_review_queue_pending": not args.offline,
            "clinical_validation": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
