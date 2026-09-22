"""Retrieval statistics with explicit denominators, and grouped dataset checks."""

import math
import statistics


def retrieval_metrics(expected, actual, k=5):
    gold = set(expected)
    ids = actual[:k]
    if not gold:
        raise ValueError("Retrieval scoring needs a nonempty gold label set")
    relevance = [int(x in gold) for x in ids]
    dcg = sum(v / math.log2(i + 2) for i, v in enumerate(relevance))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return {
        "hit": int(any(relevance)),
        "recall": len(gold.intersection(ids)) / len(gold),
        "mrr": next((1 / (i + 1) for i, v in enumerate(relevance) if v), 0),
        "ndcg": dcg / ideal,
    }


def summarize(rows):
    valid = [r for r in rows if r.get("status") == "ok"]
    return {
        "total": len(rows),
        "scored": len(valid),
        "failed": len(rows) - len(valid),
        **{
            key: sum(r["metrics"][key] for r in valid) / len(rows) if rows else None
            for key in ("hit", "recall", "mrr", "ndcg")
        },
        "retrieval_p50_ms": statistics.median(r["seconds"] * 1000 for r in valid)
        if valid
        else None,
        "retrieval_p95_ms": sorted(r["seconds"] * 1000 for r in valid)[
            math.ceil(0.95 * len(valid)) - 1
        ]
        if valid
        else None,
        "context_mean_chars": statistics.mean(r["context_chars"] for r in valid)
        if valid
        else None,
    }


def choose_strategy(reports):
    if reports["baseline"]["failed"]:
        raise ValueError("Baseline incomplete; cannot select")
    eligible = {k: r for k, r in reports.items() if not r["failed"] and r["scored"]}
    best = max(
        eligible,
        key=lambda k: (
            eligible[k]["recall"],
            eligible[k]["ndcg"],
            -eligible[k]["retrieval_p95_ms"],
        ),
    )
    baseline = reports["baseline"]
    candidate = eligible[best]
    # A latency-only tie is not a quality improvement; keep the simplest baseline.
    return (
        best
        if (candidate["recall"], candidate["ndcg"])
        > (baseline["recall"], baseline["ndcg"])
        else "baseline"
    )
