import pytest
from src.evaluation.human_review import apply_decisions,item_digest


def fixture():
    rows=[{'question_id':'q1','question':'synthetic','expected_answer':'fixture','expected_evidence_ids':['e1'],
           'reviewed':False,'metadata':{'split':'eval'}}]
    payload={'schema_version':1,'actor':'human','dataset_sha256':'digest','decisions':[
        {'question_id':'q1','item_sha256':item_digest(rows[0]),'decision':'accept',
         'source_checked':True,'reference_checked':True,'labels_checked':True}]}
    return rows,payload


def test_human_export_creates_reviewed_copy_without_mutating_original():
    rows,payload=fixture()
    result=apply_decisions(rows,payload,'digest')
    assert result[0]['reviewed'] is True
    assert rows[0]['reviewed'] is False
    assert 'human_review' not in rows[0]['metadata']


def test_model_cannot_be_imported_as_human_review():
    rows,payload=fixture();payload['actor']='model'
    with pytest.raises(ValueError):apply_decisions(rows,payload,'digest')


def test_stale_reference_or_missing_checks_cannot_be_accepted():
    rows,payload=fixture();rows[0]['expected_answer']='changed'
    with pytest.raises(ValueError):apply_decisions(rows,payload,'digest')
    rows,payload=fixture();payload['decisions'][0]['source_checked']=False
    with pytest.raises(ValueError):apply_decisions(rows,payload,'digest')


def test_unapproved_items_stay_pending_and_personal_metadata_is_rejected():
    rows,payload=fixture();payload['decisions']=[]
    assert apply_decisions(rows,payload,'digest')[0]['reviewed'] is False
    payload['reviewer_name']='unwanted-field'
    with pytest.raises(ValueError):apply_decisions(rows,payload,'digest')
