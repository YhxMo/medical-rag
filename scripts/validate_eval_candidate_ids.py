"""Validate that manual eval candidate evidence ids exist in evidence.jsonl."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate expected_evidence_ids in an eval candidate JSON file.")
    parser.add_argument("--evidence", default="artifacts/index/evidence.jsonl")
    parser.add_argument("--candidates", default="医学评估题目/manual_eval_candidates.json")
    args = parser.parse_args()

    evidence_ids = load_evidence_ids(Path(args.evidence))
    candidates = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    missing: list[tuple[str, str]] = []
    for item in candidates:
        for evidence_id in item.get("expected_evidence_ids", []):
            if evidence_id not in evidence_ids:
                missing.append((item.get("question_id", "<missing question_id>"), evidence_id))

    if missing:
        print(f"Missing {len(missing)} expected_evidence_ids:")
        for question_id, evidence_id in missing[:20]:
            print(f"- {question_id}: {evidence_id}")
        return 1

    print(f"Validated {len(candidates)} candidates against {len(evidence_ids)} evidence ids.")
    return 0


def load_evidence_ids(path: Path) -> set[str]:
    evidence_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            evidence_ids.add(json.loads(line)["evidence_id"])
    return evidence_ids


if __name__ == "__main__":
    raise SystemExit(main())
