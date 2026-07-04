from backend.app.experiments.experiment_service import normalize_experiment_id


def test_normalize_experiment_id_returns_none_for_none():
    assert normalize_experiment_id(None) is None

def test_normalize_experiment_id_returns_none_for_blank_string():
    assert normalize_experiment_id("   ") is None

def test_normalize_experiment_id_strips_value():
    assert normalize_experiment_id("  exp_123  ") == "exp_123"