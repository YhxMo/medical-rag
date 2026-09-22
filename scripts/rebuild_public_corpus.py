"""Rebuild a versioned local English index; no chat API or credentials needed."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.cli import _load_evidence, _build_embedding_provider
from src.config.settings import load_settings
from src.indexer.common import write_evidence
from src.indexer.factory import create_index_store


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_sources(evidence, per_document=20):
    plan = []
    for source in sorted({e.source_file for e in evidence}):
        pages = {}
        max_page = max(e.page_end or 0 for e in evidence if e.source_file == source)
        for item in evidence:
            if item.source_file != source or not (16 <= (item.page_start or 0) <= int(max_page * .85)):
                continue
            text = item.content.lower()
            if len(text.split()) < 90 or any(x in text for x in (
                'copyright', 'bibliography', 'contents', 'isbn', 'http', '@', 'references', 'acknowledg'
            )):
                continue
            previous = pages.get(item.page_start)
            if previous is None or len(item.content) > len(previous.content):
                pages[item.page_start] = item
        candidates = [pages[p] for p in sorted(pages)]
        if len(candidates) < per_document:
            raise ValueError('Insufficient eligible source pages')
        for i in range(per_document):
            item = candidates[int((i + .5) * len(candidates) / per_document)]
            plan.append({'sample_id': f's{len(plan)+1:03d}', 'evidence_id': item.evidence_id,
                         'source_file': source, 'page': item.page_start,
                         'split': 'dev' if i % 2 == 0 else 'eval'})
    return plan


def main():
    import os
    os.chdir(ROOT)
    started = time.monotonic()
    settings = load_settings('config.public-corpus.yaml')
    index = Path(settings.get('paths', 'index_dir'))
    if (index / 'build_manifest.json').exists():
        raise SystemExit('Index already built; use a new version directory for a new corpus.')
    manifest = json.loads(Path('docs/evaluation/public_corpus_v1.json').read_text())
    expected_files = {Path(doc['file']).name for doc in manifest['documents']}
    actual_files = {p.name for p in Path('data/raw').glob('*.pdf')
                    if p.is_file() and not p.name.startswith('.')}
    if actual_files != expected_files:
        raise ValueError('PDF collection differs from the versioned source manifest')
    for doc in manifest['documents']:
        if digest(Path(doc['file'])) != doc['sha256']:
            raise ValueError('Source checksum mismatch')
    evidence = _load_evidence(settings, image_mode='none', captioner_name='stub', ocr_mode='never')
    if len({e.evidence_id for e in evidence}) != len(evidence):
        raise ValueError('Duplicate evidence IDs')
    index.mkdir(parents=True, exist_ok=True)
    write_evidence(index / 'evidence.jsonl', evidence)
    plan = select_sources(evidence)
    plan_path = Path('data/evaluation/public-v1/sampling_plan.json')
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Text evidence prepared: {len(evidence)}; source samples: {len(plan)}', flush=True)
    provider = _build_embedding_provider('fastembed', settings)
    store = create_index_store(settings)
    try:
        store.build(evidence, provider)
        hits = store.search('What determines axial resolution in ultrasound?', provider, final_top_k=5)
        if len(hits) != 5:
            raise ValueError('Index smoke query returned insufficient results')
    finally:
        client = getattr(store, '_client', None)
        if client is not None:
            client.close()
    summary = {
        'corpus_version': 'public-medical-imaging-v1', 'embedding_model': provider.model_name,
        'embedding_dimensions': 384, 'store': 'Qdrant local + BM25/RRF',
        'chunk_size': 1200, 'chunk_overlap': 160, 'text_only': True,
        'evidence_count': len(evidence), 'source_counts': dict(Counter(e.source_file for e in evidence)),
        'source_sha256': {Path(d['file']).name: d['sha256'] for d in manifest['documents']},
        'evidence_sha256': digest(index / 'evidence.jsonl'),
        'sampling_plan_sha256': digest(plan_path),
        'embedding_artifact_sha256': {p.name: digest(p) for p in Path('models/fastembed').rglob('*')
                                     if p.is_file() and p.suffix in {'.onnx', '.json'} and not p.name.startswith('.')},
        'build_seconds': round(time.monotonic() - started, 3),
        'smoke_query_hit_count': len(hits), 'external_chat_calls': 0,
    }
    (index / 'build_manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    Path('docs/evaluation/public_index_v1.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Index ready: {len(evidence)} evidence records; 384-dimensional local embeddings.', flush=True)


if __name__ == '__main__':
    main()
