"""Conservative evidence triage, not a medical correctness classifier."""
import re


def evidence_flags(question, sources, *, initial_pass=None, audit=None):
    text='\n'.join(sources)
    flags=[]
    # Captions alone do not imply an unanswerable question. Require a request
    # for curve behaviour and a figure/curve signal in the labeled source.
    trend=bool(re.search(r'\b(vary|variation|trend|curve|relationship|dependence)\b',question,re.I))
    visual=bool(re.search(r'\b(fig(?:ure)?\.?\s*\d|curves?|graph|plot(?:ted)?)\b',text,re.I))
    if trend and visual:
        flags.append('possible_visual_dependency')
    if initial_pass is False:
        flags.append('initial_screen_failed')
        if audit and audit.get('verdict')=='supported':
            flags.append('source_review_disagreement')
    if audit and (audit.get('verdict')!='supported' or
                  audit.get('question_supported') is not True or
                  audit.get('answer_supported') is not True):
        flags.append('source_audit_not_supported')
    return flags


def assess_gate(flags, *, assistant_finding=None, answer=None):
    reasons=list(flags)
    if assistant_finding:
        reasons.append('unresolved_assistant_finding')
    if answer is not None:
        judge=answer.get('judge')
        if answer.get('status')!='ok' or not judge:
            reasons.append('missing_valid_judge')
        elif any(judge.get(k)!=2 for k in ('correctness','completeness','groundedness')):
            reasons.append('answer_quality_issue')
    return {'status':'needs_review' if reasons else 'no_flag_detected',
            'reasons':sorted(set(reasons)),
            'human_reviewed':False,
            'meaning':'triage_only_not_accuracy_or_human_approval'}
