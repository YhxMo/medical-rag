"""Create a local-only evidence viewer with optional explicit human decisions."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluation.human_review import item_digest


def main():
    base=ROOT/'data/evaluation/public-v1/all_candidates.json'
    rows=json.loads(base.read_text())
    source=ROOT/'data/evaluation/answer-quality-v1'
    audits={r['id']:r for r in json.loads((source/'ai_source_review.json').read_text())}
    answers={r['question_id']:r for r in json.loads((source/'answers.json').read_text())}
    findings_path=source/'assistant_findings.json'
    findings={r['id']:r for r in json.loads(findings_path.read_text())['findings']} if findings_path.exists() else {}
    evidence=[json.loads(line) for line in (ROOT/'artifacts/public-v1/index/evidence.jsonl').read_text().splitlines() if line]
    by_id={e['evidence_id']:e for e in evidence}
    def brief(e,gold=False):return {'file':e['source_file'],'page':e['page_start'],'text':e['content'],'gold':gold}
    draft_path=ROOT/'data/evaluation/public-v2-draft/all_candidates.json'
    drafts={q['question_id']:q for q in json.loads(draft_path.read_text())} if draft_path.exists() else {}
    triage_path=ROOT/'data/evaluation/public-v2-draft/baseline_triage.json'
    triage={q['id']:q for q in json.loads(triage_path.read_text())} if triage_path.exists() else {}
    medical_dir=ROOT/'data/evaluation/medical-review-v3'
    medical_reviews={q['question_id']:q for q in json.loads((medical_dir/'review_ledger.json').read_text())} if (medical_dir/'review_ledger.json').exists() else {}
    medical_candidates={q['question_id']:q for q in json.loads((medical_dir/'all_candidates.json').read_text())} if (medical_dir/'all_candidates.json').exists() else {}
    items=[]
    for q in rows:
        ids=set(q['expected_evidence_ids'])
        gold=[by_id[x] for x in ids]
        neighbors=[e for e in evidence if e['evidence_id'] not in ids and any(e['source_file']==g['source_file'] and abs(e['page_start']-g['page_start'])<=1 for g in gold)]
        answer=answers.get(q['question_id'])
        items.append({'id':q['question_id'],'question':q['question'],'reference':q['expected_answer'],
                      'split':q['metadata']['split'],'item_sha256':item_digest(q),
                      'original_pass':q['metadata']['automated_check_passed'],'audit':audits.get(q['question_id']),
                      'assistant_finding':findings.get(q['question_id']),
                      'triage':triage.get(q['question_id']),
                      'medical_review':medical_reviews.get(q['question_id']),
                      'medical_candidate':medical_candidates.get(q['question_id']),
                      'draft':drafts.get(q['question_id']) if drafts.get(q['question_id'],{}).get('metadata',{}).get('modified_from_frozen_baseline') else None,
                      'sources':[brief(e,True) for e in gold]+[brief(e) for e in neighbors],
                      'answer':answer,'retrieved':[brief(by_id[e]) for e in answer['retrieved_evidence_ids']] if answer else []})
    payload={'dataset_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'items':items}
    encoded=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    text=(ROOT/'scripts/templates/review.html').read_text().replace('__DATA__',encoded)
    (source/'review.html').write_text(text)
    print('Local review packet created; 60 questions; no human decisions prefilled.')
if __name__=='__main__':main()
