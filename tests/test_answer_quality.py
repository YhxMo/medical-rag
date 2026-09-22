import pytest
from src.evaluation.answer_quality import citation_check,validate_judge,aggregate_answers


def test_citation_ids_are_checked_without_claiming_semantic_support():
    result=citation_check('Claim [1] [1] and unsupported [9].',2)
    assert result['citation_mentions']==3
    assert result['valid_citation_mentions']==2
    assert result['invalid_citations']==[9]
    assert citation_check('No citation',2)['citation_id_precision'] is None


def valid_judge():
    return dict(correctness=2,completeness=2,groundedness=2,reference_supported=True,
                refusal=False,unsupported_claims=False,
                citations=[{'id':1,'supports_associated_claim':True}],reason='Supported by fixture')


@pytest.mark.parametrize('field,value',[('correctness',True),('groundedness',3),('reference_supported','true')])
def test_invalid_judge_scores_rejected(field,value):
    raw=valid_judge(); raw[field]=value
    with pytest.raises(ValueError): validate_judge(raw,[1])


def test_judge_must_assess_all_citations():
    with pytest.raises(ValueError): validate_judge(valid_judge(),[1,2])


def test_missing_answers_do_not_disappear_from_denominator():
    row={'status':'ok','judge':valid_judge(),'citations':citation_check('answer [1]',1),'answer_latency_seconds':2}
    result=aggregate_answers([row],3)
    assert result['strict_pass_count']==1
    assert result['expected_count']==3
    assert result['failed_or_unavailable_count']==2


def test_uncited_answers_do_not_pass_strict_quality():
    j=valid_judge();j['citations']=[]
    row={'status':'ok','judge':j,'citations':citation_check('answer',1),'answer_latency_seconds':1}
    result=aggregate_answers([row],1)
    assert result['strict_pass_count']==0
    assert result['semantic_citation_precision'] is None
