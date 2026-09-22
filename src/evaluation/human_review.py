"""Apply explicit, anonymous human decisions without upgrading model reviews."""
import hashlib
import json


def item_digest(row):
    return hashlib.sha256(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def apply_decisions(rows, exported, dataset_sha256):
    if not isinstance(exported,dict) or set(exported)!={'schema_version','actor','dataset_sha256','decisions'}:
        raise ValueError('Unexpected review fields')
    if exported['schema_version']!=1 or exported['actor']!='human' or exported['dataset_sha256']!=dataset_sha256:
        raise ValueError('Review actor or dataset version mismatch')
    decisions=exported['decisions']
    if not isinstance(decisions,list): raise ValueError('Decisions must be a list')
    known={q['question_id']:q for q in rows}
    mapped={}
    for d in decisions:
        if not isinstance(d,dict) or set(d)!={'question_id','item_sha256','decision','source_checked','reference_checked','labels_checked'}:
            raise ValueError('Unexpected decision fields')
        qid=d['question_id']
        if qid not in known or qid in mapped or d['item_sha256']!=item_digest(known[qid]):
            raise ValueError('Stale, duplicate or unknown review item')
        if d['decision'] not in {'accept','revise','specialist'}:
            raise ValueError('Invalid human decision')
        if any(type(d[k]) is not bool for k in ('source_checked','reference_checked','labels_checked')):
            raise ValueError('Human checks must be explicit booleans')
        if d['decision']=='accept' and not all(d[k] for k in ('source_checked','reference_checked','labels_checked')):
            raise ValueError('Accept requires all three human checks')
        mapped[qid]=d
    output=[]
    for q in rows:
        copied={**q,'metadata':dict(q.get('metadata',{}))}
        d=mapped.get(q['question_id'])
        # This creates a new reviewed version, never modifies the frozen baseline.
        copied['reviewed']=bool(d and d['decision']=='accept')
        copied['metadata']['human_review']={
            'actor':'human' if d else 'none','decision':d['decision'] if d else 'pending',
            'source_dataset_sha256':dataset_sha256,
            'qualification':'not_collected_not_claimed',
        }
        copied['metadata']['review_status']='human_source_reviewed' if copied['reviewed'] else 'pending_human_review'
        output.append(copied)
    return output
