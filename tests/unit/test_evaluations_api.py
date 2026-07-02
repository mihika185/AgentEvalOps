import pytest
from fastapi import HTTPException

from backend.app.api.evaluations import resolve_eval_set_path

def test_resolve_eval_set_path_accepts_file_name():
    assert (
        resolve_eval_set_path("agent_eval_set.json")
        == "data/eval_sets/agent_eval_set.json"
    )

def test_resolve_eval_set_path_rejects_parent_directory():
    with pytest.raises(HTTPException) as exc:
        resolve_eval_set_path("../agent_eval_set.json")

    assert exc.value.status_code == 400

def test_resolve_eval_set_path_rejects_nested_path():
    with pytest.raises(HTTPException) as exc:
        resolve_eval_set_path("nested/agent_eval_set.json")

    assert exc.value.status_code == 400