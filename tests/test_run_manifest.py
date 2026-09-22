import pytest
from src.evaluation.run_manifest import freeze_run_manifest


def test_resume_same_inputs_and_reject_changed_prompt(tmp_path):
    freeze_run_manifest(tmp_path,{'prompt':'one'})
    freeze_run_manifest(tmp_path,{'prompt':'one'})
    with pytest.raises(ValueError):freeze_run_manifest(tmp_path,{'prompt':'two'})


def test_unversioned_answers_are_not_reused(tmp_path):
    (tmp_path/'q_answer.json').write_text('{}')
    with pytest.raises(ValueError):freeze_run_manifest(tmp_path,{'prompt':'one'})
