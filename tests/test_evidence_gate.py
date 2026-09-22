from src.evaluation.evidence_gate import evidence_flags,assess_gate


def test_curve_question_on_caption_is_flagged():
    assert 'possible_visual_dependency' in evidence_flags(
        'How does efficiency vary with energy?', ['FIG. 7.1. Curves for three thicknesses.'])


def test_caption_assumptions_question_is_not_automatically_visual():
    assert evidence_flags('Which thicknesses and packing fraction are assumed?',
                          ['FIG. 7.1. Curves assume 100% packing.'])==[]


def test_textual_variation_without_figure_is_not_flagged():
    assert evidence_flags('How does attenuation vary with thickness?',
                          ['Attenuation increases with thickness.'])==[]


def test_later_pass_cannot_erase_earlier_failure():
    flags=evidence_flags('Question', ['Evidence'],initial_pass=False,
                        audit={'verdict':'supported','question_supported':True,'answer_supported':True})
    assert 'source_review_disagreement' in flags
    assert assess_gate(flags)['status']=='needs_review'


def test_assistant_finding_overrides_full_model_scores_without_changing_scores():
    answer={'status':'ok','judge':dict(correctness=2,completeness=2,groundedness=2)}
    assert assess_gate([],assistant_finding={'status':'uncertain'},answer=answer)['status']=='needs_review'
    assert answer['judge']['completeness']==2


def test_absence_of_flags_does_not_claim_approval():
    result=assess_gate([])
    assert result['status']=='no_flag_detected'
    assert result['human_reviewed'] is False


def test_inconsistent_source_audit_is_flagged():
    assert 'source_audit_not_supported' in evidence_flags('Q',['S'],audit={
        'verdict':'supported','question_supported':False,'answer_supported':True})
