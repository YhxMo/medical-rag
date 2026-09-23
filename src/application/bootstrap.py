"""Construct dependencies from one config; domain code never imports the CLI."""

from __future__ import annotations

from src.application.client import ModelClient
from src.application.service import QueryService
from src.embedding.fastembed_provider import FastEmbedProvider
from src.indexer.qdrant_store import QdrantIndexStore
from src.reranker.base import NoopReranker
from src.reranker.bge import BGEReranker


def create_store(settings):
    root = settings.resolve_path("paths", "artifact_dir", default="artifacts/rag")
    return QdrantIndexStore(root / "index", local_path=root / "qdrant")


def create_embedding(settings):
    return FastEmbedProvider(
        model_name=settings.get("embedding", "model", default="BAAI/bge-small-en-v1.5"),
        cache_dir=str(settings.resolve_path("embedding", "cache_dir", default="models/fastembed")),
    )


def create_clients(settings):
    return tuple(
        ModelClient(settings.get("models", name, default={})) for name in ("text", "vision")
    )


def build_service(settings):
    reranker = NoopReranker()
    if settings.get("reranker", "enabled", default=True):
        model = settings.get("reranker", "model", default="BAAI/bge-reranker-base")
        local = settings.path.parent / model
        reranker = BGEReranker(
            str(local) if local.exists() else model,
            device=settings.get("reranker", "device", default="cpu"),
        )
    return QueryService(
        create_store(settings),
        create_embedding(settings),
        reranker,
        *create_clients(settings),
        candidate_k=settings.get("retrieval", "candidate_k", default=20),
        top_k=settings.get("retrieval", "top_k", default=5),
        max_chars=settings.get("retrieval", "max_context_chars", default=6000),
        translate_queries=settings.get("retrieval", "translate_queries", default=True),
    )
