"""Offline triage + separate AI draft amendments. No API or baseline mutation."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluation.evidence_gate import evidence_flags,assess_gate


def main():
    source=ROOT/'data/evaluation/public-v1'
    quality=ROOT/'data/evaluation/answer-quality-v1'
    out=ROOT/'data/evaluation/public-v2-draft'
    rows=json.loads((source/'all_candidates.json').read_text())
    audits={x['id']:x for x in json.loads((quality/'ai_source_review.json').read_text())}
    answers={x['question_id']:x for x in json.loads((quality/'answers.json').read_text())}
    findings={x['id']:x for x in json.loads((quality/'assistant_findings.json').read_text())['findings']}
    evidence={x['evidence_id']:x for x in map(json.loads,(ROOT/'artifacts/public-v1/index/evidence.jsonl').read_text().splitlines())}
    hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('all_candidates.json','dev.json','eval.json')}
    checks=[]
    for q in rows:
        flags=evidence_flags(q['question'],[evidence[e]['content'] for e in q['expected_evidence_ids']],
             initial_pass=q['metadata']['automated_check_passed'],audit=audits.get(q['question_id']))
        checks.append({'id':q['question_id'],'split':q['metadata']['split'],
                       **assess_gate(flags,assistant_finding=findings.get(q['question_id']),answer=answers.get(q['question_id']))})
    changes={
        'public_v1_s006':{
            'question':'According to the caption and labels of Figure 7.1 in the IAEA handbook, which detector thicknesses and material packing fraction are assumed, and which interaction coefficient is used for the curves?',
            'expected_answer':'The thicknesses are 0.2, 0.5, and 1.0 mm with an assumed 100% material packing fraction. The curves use the photoelectric coefficient only for the primary interaction.'},
        'public_v1_s052':{
            'question':'According to the 2018 MSF ultrasound manual, how do the wall boundaries used in the clinical definition of AAA differ from the ultrasound measurement convention described there, and what reason does the manual give?',
            'expected_answer':'The manual defines AAA as an abdominal aorta greater than 3 cm measured outer wall to outer wall in true transverse. It describes ultrasound measurement as inner wall to inner wall in the anterior-posterior dimension, because the outer wall margin, particularly posteriorly next to the spine, is difficult to distinguish. These are the distinct conventions stated in this manual.'},
        'public_v1_s060':{
            'expected_answer':'The manual describes the lung point as a location where lung and air can be visualized in the same view. When scanning from anterior to lateral, a pneumothorax pattern gives way to a fleeting appearance of lung pattern at a particular chest-wall location.'},
    }
    drafts=[]
    for q in rows:
        d=copy.deepcopy(q)
        d.update(changes.get(q['question_id'],{}))
        d['reviewed']=False
        d['metadata'].update({'dataset_version':'public-v2-draft','review_status':'AI_draft_pending_independent_review',
                            'baseline_question_id':q['question_id'],'modified_from_frozen_baseline':q['question_id'] in changes,
                            'evaluation_use':'regression_only_not_independent_holdout'})
        # Old automated judgments belong to the old wording, not the draft.
        if q['question_id'] in changes:
            for key in ('automated_check','automated_check_passed'):
                d['metadata'].pop(key,None)
            d['metadata']['revision_author_type']='AI_assistant'
        drafts.append(d)
    out.mkdir(parents=True,exist_ok=True)
    def write(path,obj):
        text=json.dumps(obj,ensure_ascii=False,indent=2)+'\n'
        if path.exists() and path.read_text()!=text:
            raise ValueError('Refusing to overwrite different draft; choose a new version')
        path.write_text(text)
    for name,content in [('all_candidates',drafts),('dev',[q for q in drafts if q['metadata']['split']=='dev']),('eval',[q for q in drafts if q['metadata']['split']=='eval']),('baseline_triage',checks)]:
        write(out/f'{name}.json',content)
    assert all(hashlib.sha256((source/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())
    summary={'schema_version':1,'mode':'offline_no_API','baseline_count':len(rows),'draft_count':len(drafts),
        'modified_draft_count':len(changes),'human_reviewed_count':0,'api_calls':0,'additional_cost_cny':0,
        'gate_status_counts':dict(Counter(x['status'] for x in checks)),
        'gate_reason_counts':dict(Counter(reason for x in checks for reason in x['reasons'])),
        'baseline_sha256':hashes,'draft_sha256':hashlib.sha256((out/'all_candidates.json').read_bytes()).hexdigest(),
        'limits':'heuristic triage, not medical accuracy; draft is no longer an independent holdout; no new answer scores',
        'code_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('src/evaluation/evidence_gate.py','scripts/build_evidence_revision.py')}}
    write(ROOT/'docs/evaluation/evidence_revision_v2.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
