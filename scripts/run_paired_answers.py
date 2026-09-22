"""Frozen v3 baseline-5 vs ranked-10 answers; real calls, same-model proxy judge."""
import json,os,sys,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts.generate_public_questions import BudgetClient,read_runtime_key
from scripts.run_answer_quality import JUDGE_SYSTEM,ANSWER_SYSTEM
from src.evaluation.answer_quality import citation_check,validate_judge,aggregate_answers
from src.generator.generator import _build_citation_prompt
from src.indexer.common import read_evidence
from src.schema import RetrievalHit
from src.evaluation.run_manifest import freeze_run_manifest

OUT=ROOT/'data/evaluation/paired-answers-v1'
JUDGE=JUDGE_SYSTEM+' Score completeness against every requested part of the QUESTION, not merely similarity to the reference. A source caption does not provide curve trends. Judge only source-limited educational claims, not clinical validity. Acknowledging missing evidence does not make an incomplete answer complete.'

def main():
 os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True)
 questions=json.loads(Path('data/evaluation/medical-review-v3/regression_candidates.json').read_text())
 details=json.loads(Path('data/evaluation/context-experiment-v1/details.json').read_text())
 contexts={(r['id'],r['method']):r['evidence_ids'] for r in details}
 evidence={e.evidence_id:e for e in read_evidence(Path('artifacts/public-v1/index/evidence.jsonl'))}
 freeze_run_manifest(OUT,{'questions_sha256':hashlib.sha256(Path('data/evaluation/medical-review-v3/regression_candidates.json').read_bytes()).hexdigest(),
  'contexts_sha256':hashlib.sha256(Path('data/evaluation/context-experiment-v1/details.json').read_bytes()).hexdigest(),
  'index_sha256':hashlib.sha256(Path('artifacts/public-v1/index/evidence.jsonl').read_bytes()).hexdigest(),
  'judge_sha256':hashlib.sha256(JUDGE.encode()).hexdigest(),'answer_system':ANSWER_SYSTEM,
  'model':'deepseek-flash','temperature':.1,'max_answer_tokens':2048,'max_judge_tokens':2000})
 client=BudgetClient(read_runtime_key(),Path('experiments/results/paired-answers-v1/usage.json'),limit=14.)
 def run(q,method):
  path=OUT/(q['question_id']+'_'+method+'.json')
  if path.exists():return json.loads(path.read_text())
  ids=contexts[q['question_id'],method];hits=[RetrievalHit(evidence[e],0,i+1,{}) for i,e in enumerate(ids)]
  saved=path.with_name(path.stem+'_answer.json')
  if saved.exists():row=json.loads(saved.read_text())
  else:
   start=time.perf_counter()
   text=client.call(ANSWER_SYSTEM,_build_citation_prompt(q['question'],hits),max_tokens=2048,json_output=False,temperature=.1)
   row={'question_id':q['question_id'],'method':method,'split':q['metadata']['split'],'answer':text,'answer_latency_seconds':round(time.perf_counter()-start,4),'citations':citation_check(text,len(ids)),'retrieved_evidence_ids':ids}
   saved.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
  payload={'question':q['question'],'reference_answer':q['expected_answer'],'gold_sources':[evidence[e].content for e in q['expected_evidence_ids']],
   'retrieved_sources':[{'citation_id':i+1,'source':h.evidence.content} for i,h in enumerate(hits)],'actual_answer':row['answer'],
   'required_citation_ids':row['citations']['unique_citations'],'citation_output_rule':'Exactly one entry per required integer citation ID. No duplicates or additional IDs.'}
  for attempt in range(2):
   raw=client.call(JUDGE,payload,max_tokens=2000,temperature=.1)
   (OUT/(path.stem+f'_judge_{attempt}.json')).write_text(json.dumps(raw,ensure_ascii=False,indent=2)+'\n')
   try:row['judge']=validate_judge(raw,row['citations']['unique_citations']);break
   except ValueError:
    if attempt:raise
  row['status']='ok';path.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');print('Saved',q['question_id'],method,flush=True);return row
 jobs=[(q,m) for q in questions for m in ('baseline_5','ranked_10')]
 failures=[];results=[]
 # One authenticated job before concurrent requests; no key or exception bodies logged.
 results.append(run(*jobs[0]))
 with ThreadPoolExecutor(max_workers=3) as pool:
  futures={pool.submit(run,*job):job for job in jobs[1:]}
  for future in as_completed(futures):
   q,m=futures[future]
   try:results.append(future.result())
   except Exception as exc:failures.append({'id':q['question_id'],'method':m,'error_type':type(exc).__name__});print('Failed',q['question_id'],m,type(exc).__name__,flush=True)
 (OUT/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
 report={'same_model_judge':True,'human_reviewed':False,'budget_cny':14,'usage':client.report,'failed_jobs':len(failures),'runs':{},'question_sha256':hashlib.sha256(Path('data/evaluation/medical-review-v3/regression_candidates.json').read_bytes()).hexdigest(),'judge_prompt_sha256':hashlib.sha256(JUDGE.encode()).hexdigest()}
 for split in ('dev','eval'):
  report['runs'][split]={}
  for method in ('baseline_5','ranked_10'):
   report['runs'][split][method]=aggregate_answers([r for r in results if r['split']==split and r['method']==method],sum(q['metadata']['split']==split for q in questions))
 Path('docs/evaluation/paired_answers_v1.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'completed':len(results),'failed':len(failures),'cost_upper':client.report['known_usage_peak_cost_upper_cny']}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:print('Stopped:',type(e).__name__,flush=True);sys.exit(1)
