"""Explicit provisional retrieval checks on model-generated, unreviewed questions."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.cli import _build_embedding_provider
from src.config.settings import load_settings
from src.evaluation.dataset import load_dataset
from src.evaluation.evaluator import evaluate_retrieval, result_summary
from src.indexer.factory import create_index_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-unreviewed', action='store_true', required=True,
                        help='Explicitly acknowledge these are model-generated proxy labels, not a human benchmark')
    parser.parse_args()
    os.chdir(ROOT)
    settings = load_settings('config.public-corpus.yaml')
    store = create_index_store(settings)
    provider = _build_embedding_provider('fastembed', settings)
    index_manifest = json.loads(Path('artifacts/public-v1/index/build_manifest.json').read_text())
    evidence_ids = {e.evidence_id for e in store.load_evidence()}
    base = Path('data/evaluation/public-v1')
    report = {
        'status': 'provisional_unreviewed_source_conditioned_proxy',
        'warning': 'Not clinical accuracy or an expert benchmark; source labels are not exhaustive. No tuning performed on eval split.',
        'metrics_version': 'retrieval-v2', 'privacy_mode': 'aggregate_only',
        'index_evidence_sha256': index_manifest['evidence_sha256'],
        'embedding_model': provider.model_name, 'embedding_dimensions': 384,
        'top_k': 5, 'candidate_top_k': 20, 'splits': {},
    }
    options = {
        'bm25_only': {'dense_top_k': 0, 'bm25_top_k': 20, 'final_top_k': 20},
        'dense_only': {'dense_top_k': 20, 'bm25_top_k': 0, 'final_top_k': 20},
        'hybrid_rrf': {'dense_top_k': 20, 'bm25_top_k': 20, 'final_top_k': 20},
    }
    try:
        for split in ('dev', 'eval'):
            path = base / f'{split}.json'
            questions = load_dataset(path, reviewed_only=False)
            if any(eid not in evidence_ids for q in questions for eid in q.expected_evidence_ids):
                raise ValueError('Labels refer to missing index evidence')
            if any(q.metadata.get('split') != split for q in questions):
                raise ValueError('Split metadata mismatch')
            rows = []
            for name, kwargs in options.items():
                result = evaluate_retrieval(questions, store, provider, top_k=5, search_kwargs=kwargs)
                rows.append({'retrieval': name, **result_summary(result)})
                print(f'{split}/{name}: {result.hit_count}/{result.total} source-label hits at 5', flush=True)
            report['splits'][split] = {
                'question_count': len(questions), 'human_reviewed_count': sum(q.reviewed for q in questions),
                'automated_check_passed_count': sum(q.metadata['automated_check_passed'] for q in questions),
                'source_counts': dict(Counter(q.metadata['source_file'] for q in questions)),
                'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'all_candidates_included': True, 'runs': rows,
            }
    finally:
        if store._client is not None:
            store._client.close()
    out = Path('experiments/results/public-v1/provisional_retrieval.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n')
    Path('docs/evaluation/public_v1_provisional_retrieval.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Aggregate-only provisional report saved.', flush=True)


if __name__ == '__main__':
    main()
