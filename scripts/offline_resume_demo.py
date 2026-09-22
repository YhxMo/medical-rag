"""A runnable, self-authored miniature corpus. No API, download or textbook required."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
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
        root = Path(directory)
        store = QdrantIndexStore(
            root / "index", collection_name="synthetic_demo", local_path=root / "qdrant"
        )
        embedding = HashEmbeddingProvider()
        evidence = [
            EvidenceItem(
                "demo_1",
                "text",
                "self_authored_demo",
                "This demonstration explains provenance. A retrieval result retains its source name and page number. "
                "Citations point back to that source. These are software demonstration notes, not medical facts.",
                1,
                1,
            ),
            EvidenceItem(
                "demo_2",
                "text",
                "self_authored_demo",
                "A screenshot is a user-provided input. It is distinct from textbook evidence. "
                "Offline mode displays retrieved text without generating a model answer.",
                2,
                2,
            ),
        ]
        store.build(evidence, embedding)
        service = QueryService(
            store, embedding, NoopReranker(), None, None, version="synthetic-demo"
        )
        try:
            if args.serve:
                from src.ui.resume_app import create_resume_app

                create_resume_app(service).queue().launch(
                    server_name="127.0.0.1", share=False
                )
            else:
                print(
                    json.dumps(
                        {
                            "label": "SELF_AUTHORED_SYNTHETIC_DEMO_NOT_MEDICAL_BENCHMARK",
                            "result": service.ask(
                                "How does source provenance work?", offline=True
                            ).to_dict(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
        finally:
            if store._client:
                store._client.close()


if __name__ == "__main__":
    main()
