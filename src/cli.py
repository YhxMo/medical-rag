"""Four entry points over the same ingestion and query services."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.application.bootstrap import build_service, create_clients, create_embedding, create_store
from src.config.settings import load_settings


def main(argv=None):
    parser = argparse.ArgumentParser(description="Medical textbook RAG")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "query", "serve", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", default="config.yaml")
        if name == "index":
            sub.add_argument(
                "--captions",
                action="store_true",
                help="Call vision API for pages containing images",
            )
        elif name == "query":
            sub.add_argument("question", nargs="?", default="")
            sub.add_argument("--image")
            sub.add_argument(
                "--offline", action="store_true", help="Retrieve excerpts without model API calls"
            )
            sub.add_argument("--json", action="store_true")
        elif name == "serve":
            sub.add_argument("--port", type=int, default=7860)
        else:
            sub.add_argument("dataset", help="JSON dataset with reviewed evidence labels")
            sub.add_argument(
                "--output", help="New JSON report; defaults to experiments/runs/<timestamp>.json"
            )
    args = parser.parse_args(argv)
    store = None
    try:
        settings = load_settings(args.config)
        if args.command == "index":
            from src.document.ingest import ingest

            store = create_store(settings)
            count = ingest(
                settings.resolve_path("paths", "data_dir", default="data/raw"),
                store,
                create_embedding(settings),
                chunk_size=settings.get("chunking", "chunk_size", default=900),
                overlap=settings.get("chunking", "overlap", default=100),
                vision=create_clients(settings)[1] if args.captions else None,
            )
            print(f"Indexed {count} evidence records into {store.index_dir}")
        else:
            service = build_service(settings)
            store = service.store
            if args.command == "query":
                result = service.ask(args.question, args.image, offline=args.offline)
                print(
                    json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
                    if args.json
                    else result.answer + "\n" + "\n".join(result.warnings)
                )
            elif args.command == "serve":
                from src.ui.app import create_app

                create_app(service).queue(default_concurrency_limit=1).launch(
                    server_name="127.0.0.1", server_port=args.port, share=False
                )
            else:
                from src.evaluation.dataset import load_dataset
                from src.evaluation.evaluator import evaluate_retrieval, result_summary, save_result

                result = evaluate_retrieval(
                    load_dataset(args.dataset),
                    store,
                    service.embedding,
                    top_k=service.top_k,
                    reranker=service.reranker,
                    search_kwargs={
                        "dense_top_k": service.candidate_k,
                        "bm25_top_k": service.candidate_k,
                        "final_top_k": service.candidate_k,
                    },
                )
                output = args.output or datetime.now(timezone.utc).strftime(
                    "experiments/runs/%Y%m%dT%H%M%S.%fZ.json"
                )
                save_result(
                    output,
                    result,
                    inputs={
                        "dataset_sha256": hashlib.sha256(
                            Path(args.dataset).read_bytes()
                        ).hexdigest(),
                        "evidence_sha256": hashlib.sha256(
                            store.evidence_path.read_bytes()
                        ).hexdigest(),
                        "config": {
                            key: settings.get(key)
                            for key in ("chunking", "embedding", "reranker", "retrieval")
                        },
                    },
                )
                print(f"Saved evaluation record: {output}")
                print(json.dumps(result_summary(result), indent=2))
        return 0
    except Exception as exc:
        # Avoid dumping API request bodies or credentials; domain errors stay actionable.
        message = (
            str(exc) if isinstance(exc, (ValueError, FileNotFoundError)) else type(exc).__name__
        )
        print(f"Error: {message}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
