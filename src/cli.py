"""Command-line entry point for the medical RAG MVP."""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import sys

from src.config.settings import load_settings
from src.document.chunker import build_chunker
from src.document.exercise_parser import ExerciseParser
from src.document.image_extractor import ImageExtractionConfig, PDFImageExtractor
from src.document.loader import PDFLoader
from src.document.ocr import OCRConfig, RapidOCRPDFLoader
from src.embedding.bge import BGEEmbeddingProvider
from src.embedding.simple import HashEmbeddingProvider
from src.evaluation.dataset import load_dataset, save_dataset
from src.evaluation.evaluator import evaluate_retrieval, save_result
from src.evaluation.generator import generate_draft_questions
from src.generator.generator import DashScopeGenerator, ExtractiveGenerator
from src.indexer.evidence import build_evidence_items
from src.indexer.image_context import (
    attach_context_to_caption,
    build_contextual_caption_prompt,
    build_image_caption_context,
)
from src.indexer.factory import create_index_store
from src.reranker.base import NoopReranker, Reranker
from src.reranker.bge import BGEReranker
from src.ui.app import create_app
from src.vision.qwen_vl import QwenVLCaptioner
from src.vision.openai_vision import OpenAIVisionCaptioner
from src.vision.stub import StubVisionCaptioner


LOGGER = logging.getLogger(__name__)

_EMBEDDING_ALIASES = {
    "hash": "hash",
    "bge": "bge",
    "bge_small_zh": "bge",
}
_RERANKER_ALIASES = {
    "none": "none",
    "noop": "none",
    "bge": "bge",
    "bge_base": "bge",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medical-rag")
    parser.add_argument("--log-level", default="WARNING")
    subparsers = parser.add_subparsers(dest="command")

    for command in ("index", "serve", "evaluate", "generate-eval"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", default="config.yaml")
        if command in {"serve", "evaluate", "generate-eval"}:
            sub.add_argument("--embedding", choices=["hash", "bge"], default=None)
        if command == "index":
            sub.add_argument("--embedding", choices=["hash", "bge"], default=None)
            sub.add_argument("--image-mode", choices=["none", "ocr-pages", "all-pages"], default="ocr-pages")
            sub.add_argument("--captioner", choices=["stub", "qwen", "dashscope"], default="stub")
            sub.add_argument("--ocr", choices=["auto", "never"], default="auto")
        if command in {"evaluate", "generate-eval"}:
            sub.add_argument("--output", required=True)
        if command in {"serve", "evaluate"}:
            sub.add_argument("--reranker", choices=["none", "bge"], default=None)
        if command == "serve":
            sub.add_argument("--generator", choices=["extractive", "dashscope"], default="extractive")

    query = subparsers.add_parser("query")
    query.add_argument("question")
    query.add_argument("--config", default="config.yaml")
    query.add_argument("--embedding", choices=["hash", "bge"], default=None)
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--reranker", choices=["none", "bge"], default=None)
    query.add_argument("--generator", choices=["extractive", "dashscope"], default="extractive")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.command is None:
        parser.print_help()
        return 0

    config_path = Path(args.config)
    settings = load_settings(config_path)
    LOGGER.info("Loaded config from %s", settings.path)
    try:
        _apply_config_defaults(args, settings)
    except ValueError as exc:
        parser.error(str(exc))

    if args.command == "index":
        evidence = _load_evidence(
            settings,
            image_mode=args.image_mode,
            captioner_name=args.captioner,
            ocr_mode=args.ocr,
        )
        provider = _build_embedding_provider(args.embedding, settings)
        store = create_index_store(settings)
        try:
            store.build(evidence, provider)
        except Exception as exc:
            LOGGER.error("Failed to build index: %s", exc)
            return 1
        LOGGER.info("Indexed %d evidence items into %s", len(evidence), store.index_dir)
        return 0
    if args.command == "query":
        provider = _build_embedding_provider(args.embedding, settings)
        store = create_index_store(settings)
        try:
            hits = store.search(
                args.question,
                provider,
                dense_top_k=int(settings.get("retrieval", "hybrid", "dense_top_k", default=10)),
                bm25_top_k=int(settings.get("retrieval", "hybrid", "bm25_top_k", default=10)),
                final_top_k=args.top_k,
            )
        except Exception as exc:
            LOGGER.error("Failed to query index: %s", exc)
            return 1
        reranker = _build_reranker(args.reranker, settings)
        hits = reranker.rerank(args.question, hits, top_k=args.top_k)
        generator = _build_generator(args.generator, settings)
        answer = generator.generate(args.question, hits)
        _write_stdout(answer.answer)
        return 0
    if args.command == "serve":
        store = create_index_store(settings)
        create_app(
            store,
            embedding_factory=lambda name: _build_embedding_provider(name, settings),
            reranker_factory=lambda name: _build_reranker(name, settings),
            generator_factory=lambda name: _build_generator(name, settings),
            default_embedding=args.embedding,
            # Keep the first local-demo request fast and stable; BGE reranking
            # remains selectable from the UI for demonstrations that need it.
            default_reranker="none",
            default_generator=args.generator,
            default_top_k=int(settings.get("active", "top_k_rerank", default=5)),
        ).launch()
        return 0
    if args.command == "evaluate":
        provider = _build_embedding_provider(args.embedding, settings)
        store = create_index_store(settings)
        reranker = _build_reranker(args.reranker, settings)
        dataset_path = settings.get("evaluation", "test_dataset", default="data/test_questions.json")
        questions = load_dataset(dataset_path, reviewed_only=True)
        result = evaluate_retrieval(questions, store, provider, top_k=5, reranker=reranker)
        output_path = _resolve_output_path(args.output, "evaluation_result.json")
        save_result(output_path, result)
        LOGGER.warning(
            "Evaluated %d reviewed questions; recall@5=%.3f precision@5=%.3f mrr=%.3f ndcg@5=%.3f",
            result.total,
            result.recall_at_k,
            result.precision_at_k,
            result.mrr,
            result.ndcg_at_k,
        )
        return 0
    if args.command == "generate-eval":
        store = create_index_store(settings)
        evidence = store.load_evidence()
        questions = generate_draft_questions(evidence, limit=30)
        save_dataset(args.output, questions)
        LOGGER.warning("Wrote %d draft questions to %s", len(questions), args.output)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _build_embedding_provider(name: str, settings):
    if name == "hash":
        return HashEmbeddingProvider()
    model_name = settings.get("embedding", "bge_small_zh", "model", default="BAAI/bge-small-zh-v1.5")
    device = settings.get("embedding", "bge_small_zh", "device", default="auto")
    return BGEEmbeddingProvider(model_name=model_name, device=device)


def _build_reranker(name: str, settings) -> Reranker:
    if name == "none":
        return NoopReranker()
    model_name = settings.get("reranker", "bge_base", "model", default="BAAI/bge-reranker-base")
    device = settings.get("reranker", "bge_base", "device", default="auto")
    return BGEReranker(model_name=model_name, device=device)


def _apply_config_defaults(args, settings) -> None:
    if hasattr(args, "embedding"):
        args.embedding = _resolve_embedding_name(args.embedding, settings)
    if hasattr(args, "reranker"):
        args.reranker = _resolve_reranker_name(args.reranker, settings)


def _resolve_embedding_name(value: str | None, settings) -> str:
    configured = value or settings.get("active", "embedding", default="hash")
    normalized = str(configured).strip()
    if normalized in _EMBEDDING_ALIASES:
        return _EMBEDDING_ALIASES[normalized]
    raise ValueError(
        f"Unsupported active.embedding '{configured}'. Use one of: {', '.join(sorted(_EMBEDDING_ALIASES))}."
    )


def _resolve_reranker_name(value: str | None, settings) -> str:
    configured = value or settings.get("active", "reranker", default="none")
    normalized = str(configured).strip()
    if normalized in _RERANKER_ALIASES:
        return _RERANKER_ALIASES[normalized]
    raise ValueError(
        f"Unsupported active.reranker '{configured}'. Use one of: {', '.join(sorted(_RERANKER_ALIASES))}."
    )


def _build_generator(name: str, settings):
    if name == "extractive":
        return ExtractiveGenerator()
    return DashScopeGenerator(
        api_key=settings.get("llm", "api_key", default=""),
        api_base=settings.get("llm", "api_base", default="https://dashscope.aliyuncs.com/compatible-mode/v1"),
        model=settings.get("llm", "model", default="qwen-plus"),
        temperature=float(settings.get("llm", "temperature", default=0.1)),
        max_tokens=int(settings.get("llm", "max_tokens", default=2048)),
    )


def _build_captioner(name: str, settings):
    if name == "stub":
        return StubVisionCaptioner()
    if name == "dashscope":
        from openai import OpenAI

        api_key = settings.get("vision_captioner", "api_key", default=settings.get("llm", "api_key", default=""))
        api_base = settings.get(
            "vision_captioner",
            "api_base",
            default=settings.get("llm", "api_base", default="https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        # Pre-build the OpenAI client so every worker thread shares one
        # connection pool (keep-alive) instead of each thread opening its
        # own TLS sessions. The SDK client is thread-safe.
        client = OpenAI(api_key=api_key, base_url=api_base)
        return OpenAIVisionCaptioner(
            api_key=api_key,
            api_base=api_base,
            model=settings.get("vision_captioner", "api_model", default="qwen-vl-plus"),
            temperature=float(settings.get("vision_captioner", "temperature", default=0.1)),
            max_tokens=int(settings.get("vision_captioner", "max_tokens", default=512)),
            client=client,
        )
    return QwenVLCaptioner(
        model_name=settings.get("vision_captioner", "model", default="Qwen/Qwen3-VL-2B-Instruct")
    )


def _load_evidence(settings, *, image_mode: str, captioner_name: str, ocr_mode: str = "auto"):
    data_dir = Path(settings.get("paths", "data_dir", default="data"))
    artifact_dir = Path(settings.get("paths", "artifact_dir", default="artifacts"))
    pdf_files = sorted(data_dir.glob("*.pdf"))
    text_chunks = []
    exercise_chunks = []
    captions = []

    for pdf_path in pdf_files:
        LOGGER.info("Loading PDF: %s", pdf_path)
        pages = _load_pages(pdf_path, settings, ocr_mode=ocr_mode)
        pdf_text_chunks = []
        pdf_exercise_chunks = []
        is_exercise_book = _is_exercise_book(pdf_path)
        if is_exercise_book:
            report = ExerciseParser().parse(pages)
            pdf_exercise_chunks = list(report.chunks)
            exercise_chunks.extend(pdf_exercise_chunks)
            if report.unmatched_questions or report.unmatched_answers:
                LOGGER.warning(
                    "Exercise parse had %d unmatched questions and %d unmatched answers for %s",
                    len(report.unmatched_questions),
                    len(report.unmatched_answers),
                    pdf_path.name,
                )
        else:
            strategy = settings.active_chunking
            chunk_config = settings.get("chunking", strategy, default={})
            pdf_text_chunks = build_chunker(strategy, chunk_config).chunk(pages)
            text_chunks.extend(pdf_text_chunks)

        # Exercise/workbook PDFs in this collection carry no meaningful
        # standalone images (just question/answer text laid out on the page),
        # so skip image extraction and captioning for them entirely rather
        # than spending VLM calls on pages with nothing to describe.
        if image_mode != "none" and not is_exercise_book:
            extractor = PDFImageExtractor(
                ImageExtractionConfig(
                    output_dir=artifact_dir / "images",
                    render_page_images=True,
                    extract_embedded_images=image_mode == "all-pages",
                )
            )
            assets = extractor.extract(pdf_path, pages)
            if image_mode == "ocr-pages":
                ocr_pages = {page.page_number for page in pages if page.needs_ocr}
                assets = [asset for asset in assets if asset.kind == "page" and asset.page_number in ocr_pages]
            captioner = _build_captioner(captioner_name, settings)
            ocr_prompt = settings.get("vision_captioner", "ocr_prompt", default=None)
            caption_prompt = settings.get("vision_captioner", "caption_prompt", default=None)
            max_context_chars = int(settings.get("image_caption_context", "max_chars", default=900))
            max_context_chunks = int(settings.get("image_caption_context", "max_chunks", default=2))
            caption_cache_dir = Path(settings.get("paths", "artifact_dir", default="artifacts")) / "captions" / pdf_path.stem
            # Pre-build per-asset context and figure out which images already
            # have cached captions (loaded synchronously — cheap) vs which
            # need a live VLM call (the only expensive part worth parallelizing).
            pending: list[tuple] = []  # (asset, is_ocr, prompt, cache_path)
            for asset in assets:
                is_ocr = asset.kind == "page" and any(
                    page.page_number == asset.page_number and page.needs_ocr for page in pages
                )
                context = build_image_caption_context(
                    asset,
                    pages=pages,
                    text_chunks=pdf_text_chunks,
                    exercise_chunks=pdf_exercise_chunks,
                    max_chars=max_context_chars,
                    max_chunks=max_context_chunks,
                )
                cache_path = caption_cache_dir / f"{asset.image_id}.json"
                cached = _load_cached_caption(cache_path)
                if cached is not None:
                    captions.append(cached)
                else:
                    prompt = build_contextual_caption_prompt(ocr_prompt if is_ocr else caption_prompt, context)
                    pending.append((asset, is_ocr, prompt, cache_path, context))

            LOGGER.info(
                "Captioning %d images for %s (%d cached, %d pending)",
                len(assets), pdf_path.name, len(assets) - len(pending), len(pending),
            )

            skipped = 0
            done = len(assets) - len(pending)
            total = len(assets)
            lock = threading.Lock()

            def _caption_one(item):
                asset, is_ocr, prompt, cache_path, context = item
                return _caption_with_retry(captioner, asset, prompt=prompt, is_ocr_text=is_ocr), cache_path, context

            max_workers = int(settings.get("vision_captioner", "max_workers", default=4))
            # Clamp workers so we never spawn more threads than pending work.
            max_workers = max(1, min(max_workers, len(pending) or 1))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_caption_one, item) for item in pending]
                for future in as_completed(futures):
                    caption, cache_path, context = future.result()
                    with lock:
                        done += 1
                    if caption is None:
                        # This one image's VLM call failed after retries (bad
                        # request, transient network error, quota exhaustion,
                        # etc). Skip it rather than losing every caption
                        # already produced for this book to an unhandled
                        # exception further down the loop.
                        skipped += 1
                    else:
                        captioned = attach_context_to_caption(caption, context)
                        _save_cached_caption(cache_path, captioned)
                        captions.append(captioned)
                    with lock:
                        local_done = done
                    if local_done % 20 == 0 or local_done == total:
                        LOGGER.info(
                            "Captioned %d/%d images for %s", local_done, total, pdf_path.name
                        )
            if skipped:
                LOGGER.warning("Skipped %d/%d images for %s after retries failed", skipped, len(assets), pdf_path.name)

    return build_evidence_items(text_chunks=text_chunks, captions=captions, exercise_chunks=exercise_chunks)


def _caption_with_retry(captioner, asset, *, prompt: str | None, is_ocr_text: bool, attempts: int = 3, backoff_seconds: float = 2.0):
    """Call captioner.caption with retries so one bad image can't kill a multi-hour run."""
    for attempt in range(1, attempts + 1):
        try:
            return captioner.caption(asset, prompt=prompt, is_ocr_text=is_ocr_text)
        except Exception as exc:
            if attempt == attempts:
                LOGGER.error(
                    "Captioning failed for %s (page %s) after %d attempts: %s",
                    asset.image_id,
                    asset.page_number,
                    attempts,
                    exc,
                )
                return None
            LOGGER.warning(
                "Captioning attempt %d/%d failed for %s: %s — retrying",
                attempt,
                attempts,
                asset.image_id,
                exc,
            )
            time.sleep(backoff_seconds * attempt)
    return None


def _load_cached_caption(cache_path: Path):
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        from src.schema import ImageCaption

        return ImageCaption(**data)
    except Exception as exc:
        LOGGER.warning("Ignoring unreadable caption cache %s: %s", cache_path, exc)
        return None


def _save_cached_caption(cache_path: Path, caption) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(caption), handle, ensure_ascii=False, indent=2)


def _load_pages(pdf_path: Path, settings, *, ocr_mode: str) -> list:
    if ocr_mode == "never":
        return PDFLoader(pdf_path, ocr_min_chars=settings.ocr_min_chars).load()
    artifact_dir = Path(settings.get("paths", "artifact_dir", default="artifacts"))
    ocr_config = OCRConfig(
        cache_dir=Path(settings.get("ocr", "cache_dir", default=str(artifact_dir / "ocr"))),
        scale=float(settings.get("ocr", "scale", default=1.0)),
        min_confidence=float(settings.get("ocr", "min_confidence", default=0.5)),
        min_chars=settings.ocr_min_chars,
        use_cls=bool(settings.get("ocr", "use_cls", default=False)),
    )
    return RapidOCRPDFLoader(pdf_path, ocr_config).load()


def _is_exercise_book(path: Path) -> bool:
    return "习题" in path.stem or "学习指导" in path.stem


def _resolve_output_path(output: str, default_name: str) -> Path:
    output_path = Path(output)
    if output_path.suffix:
        return output_path
    return output_path / default_name


def _write_stdout(text: str) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
