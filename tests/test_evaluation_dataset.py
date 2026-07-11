from src.evaluation.dataset import EvaluationQuestion, load_dataset, save_dataset


def test_save_and_load_dataset_filters_reviewed(tmp_path):
    path = tmp_path / "questions.json"
    save_dataset(
        path,
        [
            EvaluationQuestion("q1", "问题1", ("e1",), reviewed=True),
            EvaluationQuestion("q2", "问题2", ("e2",), reviewed=False),
        ],
    )

    reviewed = load_dataset(path)
    all_items = load_dataset(path, reviewed_only=False)

    assert len(reviewed) == 1
    assert reviewed[0].question_id == "q1"
    assert len(all_items) == 2
