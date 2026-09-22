"""Verify metric arithmetic with public synthetic fixtures; no models or data files."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embedding.simple import HashEmbeddingProvider
from src.evaluation.dataset import EvaluationQuestion
from src.evaluation.evaluator import evaluate_retrieval, result_summary
from src.schema import EvidenceItem, RetrievalHit


class SyntheticRankingStore:
    rankings = {
        "partial": ["a"],
        "second": ["x", "b"],
        "duplicate": ["a", "a", "b", "x"],
        "outside": ["x", "y", "z", "a"],
    }

    def search(self, query, provider, **kwargs):
        return [RetrievalHit(EvidenceItem(eid, "text", "synthetic", "fixture"), 1.0, i)
                for i, eid in enumerate(self.rankings[query], start=1)]


def build_report():
    questions = [EvaluationQuestion(str(i), name, ("a", "b"), reviewed=True)
                 for i, name in enumerate(SyntheticRankingStore.rankings, start=1)]
    result = evaluate_retrieval(
        questions, SyntheticRankingStore(), HashEmbeddingProvider(), top_k=3,
        search_kwargs={"final_top_k": 4},
    )
    expected = {"recall_at_k": 0.5, "precision_at_k": 1 / 3,
                "hit_rate_at_k": 0.75, "mrr": 0.625, "ndcg_at_k": 0.5}
    for key, value in expected.items():
        if not math.isclose(getattr(result, key), value, rel_tol=1e-12):
            raise RuntimeError(f"Synthetic metric invariant failed: {key}")
    report = result_summary(result)
    report.update({
        "fixture_version": "synthetic-ranking-v1",
        "scope": "metric_arithmetic_only_not_medical_retrieval_quality",
        "python_version": platform.python_version(),
        "verification": "passed",
        "external_model_calls": 0,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("src/evaluation/evaluator.py", "src/evaluation/dataset.py",
                         "src/evaluation/ablation.py", "src/cli.py",
                         "scripts/verify_evaluation_offline.py")
        },
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_report(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Synthetic metric verification passed; aggregate-only report written.")


if __name__ == "__main__":
    main()
