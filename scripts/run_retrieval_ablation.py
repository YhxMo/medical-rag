"""Run offline retrieval ablation experiments."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import _build_embedding_provider, _build_reranker
from src.config.settings import load_settings
from src.evaluation.ablation import default_ablation_specs, run_retrieval_ablation, write_ablation_outputs
from src.evaluation.dataset import load_dataset
from src.indexer.factory import create_index_store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run retrieval ablation experiments.")
    parser.add_argument("--config", default="config.yaml", help="Path to local YAML config.")
    parser.add_argument("--dataset", default=None, help="Evaluation dataset path. Defaults to config value.")
    parser.add_argument("--index-dir", default=None, help="Index directory. Defaults to config value.")
    parser.add_argument("--output-dir", default="experiments/results", help="Directory for JSON and Markdown outputs.")
    parser.add_argument("--tmp-index-dir", default="artifacts/tmp_ablation", help="Directory for temporary indexes.")
    parser.add_argument("--embedding", choices=["hash", "bge"], default="bge", help="Embedding provider for search.")
    parser.add_argument("--reranker", choices=["none", "bge"], default="bge", help="Reranker for the reranker ablation.")
    parser.add_argument("--top-k", type=int, default=5, help="Metric cutoff.")
    parser.add_argument("--candidate-top-k", type=int, default=20, help="Candidate pool before optional reranking.")
    parser.add_argument(
        "--skip-no-image-caption",
        action="store_true",
        help="Skip the temporary-index run that removes image_caption evidence.",
    )
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    dataset_path = args.dataset or settings.get("evaluation", "test_dataset", default="data/test_questions.json")
    index_dir = args.index_dir or settings.get("paths", "index_dir", default="artifacts/index")
    questions = load_dataset(dataset_path, reviewed_only=True)
    store = create_index_store(settings, index_dir=index_dir)
    embedding_provider = _build_embedding_provider(args.embedding, settings)
    reranker = None if args.reranker == "none" else _build_reranker(args.reranker, settings)
    specs = default_ablation_specs(
        candidate_top_k=args.candidate_top_k,
        include_reranker=reranker is not None,
        include_no_image_caption=not args.skip_no_image_caption,
    )

    report = run_retrieval_ablation(
        questions,
        store,
        embedding_provider,
        reranker=reranker,
        top_k=args.top_k,
        candidate_top_k=args.candidate_top_k,
        tmp_index_dir=args.tmp_index_dir,
        specs=specs,
    )

    json_path, markdown_path = write_ablation_outputs(report, args.output_dir)
    print(f"Wrote {json_path} and {markdown_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
