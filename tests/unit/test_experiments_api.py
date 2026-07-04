from backend.app.api.experiments import (
    clean_optional_string,
    create_experiment_id,
    normalize_metadata_json,
)


def test_create_experiment_id_has_expected_prefix():
    experiment_id = create_experiment_id()

    assert experiment_id.startswith("exp_")
    assert len(experiment_id) == len("exp_") + 12


def test_clean_optional_string_trims_text():
    assert clean_optional_string("  hello  ") == "hello"


def test_clean_optional_string_returns_none_for_blank_text():
    assert clean_optional_string("   ") is None


def test_normalize_metadata_json_returns_empty_dict_for_none():
    assert normalize_metadata_json(None) == {}


def test_normalize_metadata_json_returns_copy():
    original = {"retriever": "hybrid"}

    normalized = normalize_metadata_json(original)

    assert normalized == {"retriever": "hybrid"}
    assert normalized is not original