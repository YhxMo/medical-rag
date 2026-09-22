"""Frozen retrieval selection and paired online evaluation with explicit failures."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.application.client import atomic_json, digest
from src.application.images import prepare_image
from src.application.service import build_service, pack_context, BOUNDARY
from src.config.settings import load_settings
from src.evaluation.run_manifest import freeze_run_manifest
from src.evaluation.resume_metrics import retrieval_metrics, summarize, choose_strategy
from src.evaluation.resume_reporting import operational_metrics, answer_context
from src.indexer.common import retrieval_text
from scripts.prepare_resume_v2 import sha


def close(service):
    if getattr(service.store, "_client", None) is not None:
        service.store._client.close()


def retrieval(settings, dataset, strategies, *, regression=False):
    rows = json.loads(Path(dataset).read_text())
    if not regression:
        frozen = Path(dataset).with_name("frozen.json")
        if not frozen.exists() or json.loads(frozen.read_text())[
            "dataset_sha256"
        ] != sha(dataset):
            raise ValueError("Dataset is not frozen or has changed")
    questions = (
        rows
        if regression
        else [q for q in rows if q["split"] == "dev" and q["kind"] == "text"]
    )
    root = ROOT / "artifacts/resume-v2"
    out = root / ("regression-retrieval" if regression else "dev-retrieval")
    if not regression and (root / "selection.json").exists():
        selected = json.loads((root / "selection.json").read_text())
        if selected["dataset_sha256"] != sha(dataset):
            raise ValueError("Selection is frozen against a different dataset")
        for strategy, expected_version in selected["versions"].items():
            service = build_service(settings, strategy=strategy)
            try:
                if service.version != expected_version:
                    raise ValueError(
                        "Selected index changed; use a new experiment version"
                    )
            finally:
                close(service)
        print((out / "summary.json").read_text())
        return  # Reuse completed selection; measurement timestamps are not inputs.
    details = []
    reports = {}
    versions = {}
    for strategy in strategies:
        service = build_service(settings, strategy=strategy)
        versions[strategy] = service.version
        freeze_run_manifest(
            out / strategy,
            {
                "dataset": sha(dataset),
                "index": service.version,
                "strategy": strategy,
                "top_k": 5,
                "characters": 6000,
                "regression": regression,
            },
        )
        try:
            # Separate warm-up from measured retrieval. Model initialization is not timed as steady-state.
            service.embedding.embed_query("warmup")
            if strategy == "rerank":
                warm = service.store.search("warmup", service.embedding, final_top_k=1)
                service.reranker.rerank("warmup", warm)
            current = []
            for q in questions:
                try:
                    start = time.perf_counter()
                    hits = service.store.search(
                        q["question"],
                        service.embedding,
                        dense_top_k=20,
                        bm25_top_k=20,
                        final_top_k=20,
                    )
                    hits = service.reranker.rerank(q["question"], hits, top_k=20)
                    hits = pack_context(hits)
                    seconds = time.perf_counter() - start
                    row = {
                        "id": q["question_id"],
                        "strategy": strategy,
                        "status": "ok",
                        "seconds": seconds,
                        "metrics": retrieval_metrics(
                            q["expected_evidence_ids"],
                            [h.evidence.evidence_id for h in hits],
                        ),
                        "context_chars": sum(
                            len(retrieval_text(h.evidence)) for h in hits
                        ),
                        "evidence_ids": [h.evidence.evidence_id for h in hits],
                    }
                except Exception as exc:
                    row = {
                        "id": q["question_id"],
                        "strategy": strategy,
                        "status": "failed",
                        "error": type(exc).__name__,
                    }
                current.append(row)
            reports[strategy] = summarize(current)
            details.extend(current)
        except Exception as exc:
            current = [
                {
                    "id": q["question_id"],
                    "strategy": strategy,
                    "status": "failed",
                    "error": type(exc).__name__,
                }
                for q in questions
            ]
            reports[strategy] = summarize(current)
            details.extend(current)
        finally:
            close(service)
    summary = {
        "dataset_sha256": sha(dataset),
        "kind": "contaminated_regression" if regression else "development_selection",
        "reports": reports,
        "versions": versions,
        "hardware": platform.platform(),
        "machine": platform.machine(),
        "measurement": "one warm pass; excludes model initialization; no clinical accuracy",
        "missing_strategies": sorted(
            set(("baseline", "chapter", "rerank")) - set(strategies)
        ),
    }
    atomic_json(out / "details.json", details)
    atomic_json(out / "summary.json", summary)
    if not regression:
        if set(strategies) != {"baseline", "chapter", "rerank"} or any(
            r["failed"] for r in reports.values()
        ):
            raise RuntimeError(
                "All three development strategies must complete before selection"
            )
        selection = {
            "strategy": choose_strategy(reports),
            "dataset_sha256": sha(dataset),
            "versions": versions,
            "selection_rule": "dev text Recall@5, then NDCG@5, then P95; keep baseline on quality tie",
            "development_summary_sha256": sha(out / "summary.json"),
        }
        selection_path = root / "selection.json"
        if (
            selection_path.exists()
            and json.loads(selection_path.read_text()) != selection
        ):
            raise ValueError(
                "A strategy is already frozen; use a new experiment version"
            )
        atomic_json(selection_path, selection)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


JUDGE = (
    BOUNDARY
    + """Evaluate the actual answer against original gold evidence and retrieved evidence. Return JSON:
{"correctness":0,"completeness":0,"groundedness":0,"behavior_ok":false,
"refused":false,"unsupported_claims":false,"image_claims_supported":false,
"citations":[{"id":1,"supports_associated_claim":true}],"reason":"..."}.
Scores 0/1/2. Groundedness must use only evidence available to the answering model, not gold it did not retrieve.
For missing-information questions, reward explicit refusal to invent the absent value; for false premises reward
source-supported correction. Do not reward reference wording similarity. The reference is provisional.
Account for every question part. Assess every unique cited number exactly once; invalid IDs are unsupported.
Citation numbering alone does not establish semantic support.
This is an AI proxy assessment, not expert review."""
)


def judge_valid(row, refs=()):
    if any(
        type(row.get(k)) is not int or row[k] not in (0, 1, 2)
        for k in ("correctness", "completeness", "groundedness")
    ):
        raise ValueError("Invalid judge scores")
    if any(
        type(row.get(k)) is not bool
        for k in (
            "behavior_ok",
            "refused",
            "unsupported_claims",
            "image_claims_supported",
        )
    ):
        raise ValueError("Invalid judge flags")
    if not isinstance(row.get("reason"), str):
        raise ValueError("Missing judge reason")
    citations = row.get("citations")
    if not isinstance(citations, list) or len(citations) != len(set(refs)):
        raise ValueError("Judge must assess every unique citation")
    if {c.get("id") for c in citations} != set(refs) or any(
        type(c.get("id")) is not int
        or type(c.get("supports_associated_claim")) is not bool
        for c in citations
    ):
        raise ValueError("Invalid semantic citation assessment")
    return row


def online(settings, dataset, split):
    rows = json.loads(Path(dataset).read_text())
    base = ROOT / "artifacts/resume-v2"
    if split == "holdout":
        decision_path = base / "dev_review_decision.json"
        review_path = Path(dataset).parent / "answers-dev/review_completed.json"
        if not decision_path.exists() or not review_path.exists():
            raise RuntimeError(
                "Development source review is required before the one-time holdout run"
            )
        decision = json.loads(decision_path.read_text())
        if decision["dataset_sha256"] != sha(dataset) or decision[
            "review_sha256"
        ] != sha(review_path):
            raise ValueError("Development review decision has changed inputs")
        if not (
            Path(dataset).parent / "answers-dev/judge-repair/summary.json"
        ).exists():
            raise RuntimeError(
                "Development judge repair pass must finish before holdout"
            )
    frozen = json.loads(Path(dataset).with_name("frozen.json").read_text())
    if frozen["dataset_sha256"] != sha(dataset):
        raise ValueError("Dataset changed")
    selection = json.loads((base / "selection.json").read_text())
    if selection["dataset_sha256"] != sha(dataset):
        raise ValueError("Selection uses different dataset")
    if not (base / "multimodal/build_manifest.json").exists():
        raise ValueError("Multimodal index required")
    service = build_service(settings)
    if not service.text.available or not service.vision.available:
        raise RuntimeError("Both API credentials required")
    if service.version != selection["versions"][selection["strategy"]]:
        raise ValueError("Selected index changed")
    out = ROOT / f"data/evaluation/resume-v2/answers-{split}"
    freeze_run_manifest(
        out,
        {
            "dataset": sha(dataset),
            "selection": sha(base / "selection.json"),
            "index": service.version,
            "judge": digest(JUDGE),
            "split": split,
            "text_model": service.text.config["model"],
            "vision_model": service.vision.config["model"],
        },
    )
    questions = [q for q in rows if q["split"] == split]
    gold = {e.evidence_id: e for e in service.store.load_evidence()}
    results = []
    try:
        for q in questions:
            modes = (
                ["text_only", "captions", "originals"]
                if q["kind"] == "chart"
                else ["auto"]
            )
            for mode in modes:
                path = out / f"{q['question_id']}_{mode}.json"
                if path.exists():
                    results.append(json.loads(path.read_text()))
                    continue
                result = service.ask(
                    q["question"],
                    q.get("image_path"),
                    category="evaluation",
                    modality=mode,
                    event=lambda _: None,
                )
                record = {
                    "question_id": q["question_id"],
                    "kind": q["kind"],
                    "mode": mode,
                    "expected_behavior": q["expected_behavior"],
                    "AI_source_audit": q["AI_source_audit"],
                    "result": result.to_dict(),
                    "judge": None,
                    "status": "failed",
                }
                atomic_json(
                    path, record
                )  # Failed jobs remain in the denominator; no silent regeneration.
                if result.status != "service_error":
                    payload = {
                        "question": q["question"],
                        "reference": q["reference_answer"],
                        "expected_behavior": q["expected_behavior"],
                        "answer": result.answer,
                        "gold_sources": [
                            {"id": i, "content": gold[i].content}
                            for i in q["expected_evidence_ids"]
                            if i in gold
                        ],
                        "retrieved_sources": result.to_dict()["sources"],
                        "required_citation_ids": result.citations.get(
                            "unique_citations", []
                        ),
                        "answer_received_originals": bool(result.image_paths)
                        or bool(q.get("image_path")),
                        "image_order": [
                            "gold_original_not_necessarily_seen_by_answerer"
                        ]
                        + ["answerer_retrieved_original"] * len(result.image_paths[:2]),
                    }
                    try:
                        judge, usage = service.vision.call(
                            JUDGE,
                            payload,
                            images=[prepare_image(q["gold_image"])[0]]
                            + [prepare_image(p)[0] for p in result.image_paths[:2]],
                            category="evaluation",
                            version=service.version,
                            max_tokens=1024,
                        )
                        atomic_json(
                            out / f"{q['question_id']}_{mode}_judge_raw.json", judge
                        )
                        record.update(
                            judge=judge_valid(
                                judge, result.citations.get("unique_citations", [])
                            ),
                            judge_usage=usage,
                            status="ok",
                        )
                    except Exception as exc:
                        record["judge_error"] = type(exc).__name__
                    atomic_json(path, record)
                results.append(record)
                print(
                    f"{split} {q['question_id']} {mode} {record['status']}", flush=True
                )
    finally:
        close(service)
    summary = {
        "split": split,
        "expected_runs": sum(3 if q["kind"] == "chart" else 1 for q in questions),
        "completed_answers": sum(
            r["result"]["status"] != "service_error" for r in results
        ),
        "judged": sum(r["status"] == "ok" for r in results),
        "failed": sum(r["status"] != "ok" for r in results),
        "human_reviewed": 0,
        "judge_scope": "AI proxy; visual generator/judge share model; no clinical accuracy",
        "by_mode": {},
    }
    for mode in ("auto", "text_only", "captions", "originals"):
        subset = [r for r in results if r["mode"] == mode]
        valid = [r for r in subset if r["judge"]]
        summary["by_mode"][mode] = {
            "total": len(subset),
            "judged": len(valid),
            "strict_proxy_pass": sum(
                all(
                    r["judge"][s] == 2
                    for s in ("correctness", "completeness", "groundedness")
                )
                and r["judge"]["behavior_ok"]
                and not r["judge"]["unsupported_claims"]
                and (
                    r["expected_behavior"] == "refuse"
                    or (
                        r["result"]["citations"].get("has_citation")
                        and not r["result"]["citations"].get("invalid_citations")
                        and all(
                            c["supports_associated_claim"]
                            for c in r["judge"]["citations"]
                        )
                    )
                )
                for r in valid
            ),
        }
    behaviors = [r for r in results if r["kind"] == "behavior" and r["judge"]]
    refuse = [r for r in behaviors if r["expected_behavior"] == "refuse"]
    answerable = [
        r for r in results if r["expected_behavior"] != "refuse" and r["judge"]
    ]
    summary["refusal"] = {
        "unanswerable_scored": len(refuse),
        "wrong_answers": sum(not r["judge"]["refused"] for r in refuse),
        "answerable_scored": len(answerable),
        "false_refusals": sum(r["judge"]["refused"] for r in answerable),
    }
    summary["operations_by_kind_and_mode"] = operational_metrics(results, questions)
    # Freeze a review queue: every failure, all changed ablation outcomes, plus ten deterministic random cases.
    ids = set(
        r["question_id"]
        for r in results
        if r["status"] != "ok" or not r["judge"]["behavior_ok"]
    )
    grouped = {}
    for r in results:
        grouped.setdefault(r["question_id"], []).append(r)
    for id, items in grouped.items():
        paired = {r["mode"]: r for r in items}
        if "captions" in paired and "originals" in paired:
            if answer_context(
                paired["captions"]["result"]["sources"]
            ) != answer_context(paired["originals"]["result"]["sources"]):
                raise ValueError(
                    "Image ablation did not hold retrieved context constant"
                )
        if len({json.dumps(r["judge"], sort_keys=True) for r in items}) > 1:
            ids.add(id)
    ids.update(random.Random(20260921).sample(sorted(grouped), min(10, len(grouped))))
    atomic_json(
        out / "review_queue.json",
        {"ids": sorted(ids), "status": "pending_source_review", "human_reviewed": 0},
    )
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["retrieval", "answers"])
    parser.add_argument("--dataset", default="data/evaluation/resume-v2/dataset.json")
    parser.add_argument("--regression", action="store_true")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["baseline", "chapter", "rerank"],
        default=["baseline", "chapter", "rerank"],
    )
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    args = parser.parse_args()
    settings = load_settings(ROOT / "config.resume-v2.yaml")
    if args.command == "retrieval":
        retrieval(settings, args.dataset, args.strategies, regression=args.regression)
    else:
        online(settings, args.dataset, args.split)
