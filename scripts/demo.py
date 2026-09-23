"""Run the real Qdrant/BM25/query path with synthetic data and no model download."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.application.service import QueryService
from src.embedding.simple import HashEmbeddingProvider
from src.indexer.qdrant_store import QdrantIndexStore
from src.reranker.base import NoopReranker
from src.schema import EvidenceItem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="medical-rag-demo-") as directory:
        store = QdrantIndexStore(Path(directory) / "index")
        embedding = HashEmbeddingProvider()
        evidence = [
            EvidenceItem(
                "demo-1",
                "text",
                "demo-notes.pdf",
                "Source provenance keeps the book name and PDF page number with every chunk. "
                "Citations point back to the retrieved source.",
                1,
                1,
            ),
            EvidenceItem(
                "demo-2",
                "text",
                "demo-notes.pdf",
                "Offline retrieval displays source excerpts without calling a language model.",
                2,
                2,
            ),
            EvidenceItem(
                "demo-3",
                "text",
                "demo-notes.pdf",
                "A screenshot is a user input, not a textbook source.",
                3,
                3,
            ),
        ]
        try:
            store.build(evidence, embedding)
            service = QueryService(store, embedding, NoopReranker())
            if args.serve:
                from src.ui.app import create_app

                create_app(service).queue().launch(server_name="127.0.0.1", share=False)
            else:
                result = service.ask("How does source provenance work?", offline=True)
                print(
                    json.dumps(
                        {
                            "demo": "Synthetic software notes; not a medical benchmark",
                            "result": result.to_dict(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
        finally:
            store.close()


if __name__ == "__main__":
    main()
