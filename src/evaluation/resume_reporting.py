"""Descriptive online metrics with explicit missing and failed denominators."""

from collections import Counter
import math


def answer_context(sources):
    """Evidence identity, order and exact model-visible text; scores are telemetry."""
    return [
        {k: row[k] for k in ("id", "source", "page", "type", "content")}
        for row in sources
    ]


def distribution(values):
    values = sorted(
        v for v in values if isinstance(v, (int, float)) and math.isfinite(v)
    )

    def percentile(p):
        if not values:
            return None
        index = (len(values) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return {"n": len(values), "p50": percentile(0.5), "p95": percentile(0.95)}


def operational_metrics(records, questions):
    gold = {q["question_id"]: q for q in questions}
    groups = {}
    for row in records:
        groups.setdefault(row["kind"] + "/" + row["mode"], []).append(row)
    report = {}
    for group, rows in sorted(groups.items()):
        # Cached answer replay must never be reported as fresh model latency.
        live = [
            r
            for r in rows
            if r["result"].get("calls")
            and not any(c.get("cache_hit") for c in r["result"]["calls"])
        ]
        calls = [c for r in rows for c in r["result"].get("calls", [])]
        paid = [c for c in calls if not c.get("cache_hit")]
        judges = [r["judge"] for r in rows if r.get("judge") and r["status"] == "ok"]
        source_hits = sum(
            bool(
                set(gold[r["question_id"]]["expected_evidence_ids"])
                & {s["id"] for s in r["result"]["sources"]}
            )
            for r in rows
        )
        report[group] = {
            "runs": len(rows),
            "failed": sum(r["status"] != "ok" for r in rows),
            "status_counts": dict(Counter(r["result"]["status"] for r in rows)),
            "source_hits": source_hits,
            "source_hit_denominator": len(rows),
            "judged": len(judges),
            "distinct_tasks": len({r["question_id"] for r in rows}),
            "image_claims_supported_among_judged": sum(
                j.get("image_claims_supported") is True for j in judges
            ),
            "semantic_citations_scored": sum(
                len(j.get("citations", [])) for j in judges
            ),
            "semantic_citations_supported": sum(
                c["supports_associated_claim"]
                for j in judges
                for c in j.get("citations", [])
            ),
            "score_means_among_judged": {
                k: sum(j[k] for j in judges) / len(judges) if judges else None
                for k in ("correctness", "completeness", "groundedness")
            },
            "unsupported_expansions": sum(j["unsupported_claims"] for j in judges),
            "live_answer_runs": len(live),
            "latency_seconds_live_only": {
                k: distribution(r["result"].get("timings", {}).get(k) for r in live)
                for k in (
                    "retrieval_seconds",
                    "model_seconds",
                    "total_seconds",
                    "first_token_seconds",
                    "model_first_token_seconds",
                )
            },
            "context_characters": distribution(
                sum(len(s["content"]) for s in r["result"]["sources"]) for r in rows
            ),
            "textbook_originals": distribution(
                len(r["result"].get("image_paths", [])) for r in rows
            ),
            "uploaded_images": sum(
                bool(gold[r["question_id"]].get("image_path")) for r in rows
            ),
            "cached_model_calls": len(calls) - len(paid),
            "reported_model_tokens": {
                k: sum((c.get("usage") or {}).get(k, 0) for c in paid)
                for k in ("input_tokens", "output_tokens")
            },
            "calls_with_unknown_usage": sum(c.get("usage") is None for c in paid),
            "reported_answer_cost_cny": sum(c.get("cost_upper_cny") or 0 for c in paid),
            "cost_scope": "Successful answer-call metadata only; shared ledger includes judges, failed requests and unresolved reservations.",
        }
    return report
