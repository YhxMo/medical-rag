"""Offline development experiment followed by explicitly contaminated regression."""
import hashlib,json,os,sys,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.cli import _build_embedding_provider
from src.config.settings import load_settings
from src.indexer.factory import create_index_store
from src.retrieval.context import ContextExpander


def main():
 os.chdir(ROOT)
 path=Path('data/evaluation/medical-review-v3/regression_candidates.json')
 questions=json.loads(path.read_text())
 settings=load_settings('config.public-corpus.yaml');store=create_index_store(settings)
 provider=_build_embedding_provider('fastembed',settings);expander=ContextExpander(store.load_evidence())
 output=[]
 try:
  provider.embed_query('warmup')
  for split in ('dev','eval'):
   for q in questions:
    if q['metadata']['split']!=split:continue
    start=time.perf_counter();hits=store.search(q['question'],provider,dense_top_k=20,bm25_top_k=20,final_top_k=10);elapsed=time.perf_counter()-start
    start=time.perf_counter();expanded=expander.expand(hits[:5]);extension=time.perf_counter()-start
    for name,context in [('baseline_5',hits[:5]),('ranked_10',hits),('neighbors_10',expanded)]:
     ids=[h.evidence.evidence_id for h in context];gold=set(q['expected_evidence_ids']);found=gold.intersection(ids)
     output.append({'id':q['question_id'],'split':split,'method':name,'hit':bool(found),'label_recall':len(found)/len(gold),
        'all_labels_present':gold.issubset(ids),'context_items':len(ids),'context_chars':sum(len(h.evidence.content) for h in context),
        'retrieval_seconds':elapsed,'expansion_seconds':extension if name=='neighbors_10' else 0,'evidence_ids':ids})
 finally:
  if store._client is not None:store._client.close()
 report={'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'API_calls':0,'methods_frozen_before_run':True,
 'limits':'Source-conditioned AI-reviewed regression labels, not exhaustive. Eval cases already inspected; not blind holdout. Coverage with extra context is not Recall@5 or answer accuracy. One warm run; shared search timing excludes model initialization.', 'splits':{}}
 for split in ('dev','eval'):
  report['splits'][split]={}
  for method in ('baseline_5','ranked_10','neighbors_10'):
   rs=[r for r in output if r['split']==split and r['method']==method]
   report['splits'][split][method]={'n':len(rs),'hits':sum(r['hit'] for r in rs),'mean_label_coverage':statistics.mean(r['label_recall'] for r in rs),'all_labels_present':sum(r['all_labels_present'] for r in rs),'mean_context_chars':round(statistics.mean(r['context_chars'] for r in rs)), 'median_shared_search_ms':round(statistics.median(r['retrieval_seconds'] for r in rs)*1000,2),'median_expansion_ms':round(statistics.median(r['expansion_seconds'] for r in rs)*1000,2)}
 out=Path('data/evaluation/context-experiment-v1');out.mkdir(parents=True,exist_ok=True)
 (out/'details.json').write_text(json.dumps(output,indent=2)+'\n')
 Path('docs/evaluation/context_experiment_v1.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
