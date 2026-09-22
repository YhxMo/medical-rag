"""Deterministic citation checks and strictly validated AI-judge aggregation."""
import math
import re
import statistics


def citation_check(answer, source_count):
    refs = [int(x) for x in re.findall(r'\[(\d+)\]', answer)]
    valid = [x for x in refs if 1 <= x <= source_count]
    return {
        'citation_mentions': len(refs), 'valid_citation_mentions': len(valid),
        'unique_citations': sorted(set(refs)),
        'invalid_citations': sorted(set(refs) - set(valid)),
        'has_citation': bool(refs),
        'citation_id_precision': len(valid)/len(refs) if refs else None,
    }


def validate_judge(raw, refs):
    if not isinstance(raw, dict):
        raise ValueError('Judge result must be an object')
    for field in ('correctness', 'completeness', 'groundedness'):
        if type(raw.get(field)) is not int or raw[field] not in (0,1,2):
            raise ValueError('Judge scores must be integers 0..2')
    for field in ('reference_supported', 'refusal', 'unsupported_claims'):
        if type(raw.get(field)) is not bool:
            raise ValueError('Judge flags must be booleans')
    entries = raw.get('citations')
    if not isinstance(entries, list):
        raise ValueError('Judge citation assessments missing')
    seen = set()
    clean = []
    for row in entries:
        if not isinstance(row, dict) or type(row.get('id')) is not int or row['id'] not in refs or row['id'] in seen:
            raise ValueError('Invalid judge citation mapping')
        if type(row.get('supports_associated_claim')) is not bool:
            raise ValueError('Citation support flag must be boolean')
        seen.add(row['id'])
        clean.append({'id':row['id'], 'supports_associated_claim':row['supports_associated_claim']})
    if seen != set(refs):
        raise ValueError('Judge must assess every cited ID')
    reason = raw.get('reason', '')
    if not isinstance(reason,str) or len(reason)>1800:
        raise ValueError('Invalid judge explanation')
    return {**{k:raw[k] for k in ('correctness','completeness','groundedness',
                                 'reference_supported','refusal','unsupported_claims')},
            'citations':clean, 'reason':reason}


def aggregate_answers(rows, expected_count):
    complete=[r for r in rows if r.get('status')=='ok']
    judged=[r for r in complete if r.get('judge')]
    mention_count=sum(r['citations']['citation_mentions'] for r in complete)
    valid_mentions=sum(r['citations']['valid_citation_mentions'] for r in complete)
    citations=[c for r in judged for c in r['judge']['citations']]
    latencies=[r['answer_latency_seconds'] for r in complete]
    def ratio(n,d): return n/d if d else None
    return {
        'expected_count':expected_count, 'answer_success_count':len(complete), 'judged_count':len(judged),
        'failed_or_unavailable_count':expected_count-len(complete),
        'judge_label':'same_model_AI_proxy_not_human_clinical_accuracy',
        'strict_pass_count':sum(all(r['judge'][s]==2 for s in ('correctness','completeness','groundedness'))
                                and r['judge']['reference_supported']
                                and not r['judge']['unsupported_claims']
                                and r['citations']['has_citation'] and not r['citations']['invalid_citations']
                                and all(c['supports_associated_claim'] for c in r['judge']['citations']) for r in judged),
        'correctness_full_count':sum(r['judge']['correctness']==2 for r in judged),
        'completeness_full_count':sum(r['judge']['completeness']==2 for r in judged),
        'groundedness_full_count':sum(r['judge']['groundedness']==2 for r in judged),
        'unsupported_claim_answer_count':sum(r['judge']['unsupported_claims'] for r in judged),
        'reference_disputed_count':sum(not r['judge']['reference_supported'] for r in judged),
        'refusal_count':sum(r['judge']['refusal'] for r in judged),
        'answers_with_citations':sum(r['citations']['has_citation'] for r in complete),
        'citation_mentions':mention_count, 'valid_citation_mentions':valid_mentions,
        'citation_id_precision':ratio(valid_mentions,mention_count),
        'semantic_citation_assessments':len(citations),
        'semantic_citation_supported':sum(c['supports_associated_claim'] for c in citations),
        'semantic_citation_precision':ratio(sum(c['supports_associated_claim'] for c in citations),len(citations)),
        'answer_latency_median_seconds':statistics.median(latencies) if latencies else None,
        'answer_latency_p95_seconds':sorted(latencies)[math.ceil(.95*len(latencies))-1] if latencies else None,
    }
