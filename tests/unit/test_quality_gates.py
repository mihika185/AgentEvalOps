import pytest

from backend.app.evaluation.quality_gates import compare_metric


def test_greater_than_or_equal_gate_passes_when_value_meets_threshold():
    assert compare_metric(metric_value=0.85, operator=">=", threshold=0.8) is True


def test_greater_than_or_equal_gate_fails_when_value_is_below_threshold():
    assert compare_metric(metric_value=0.75, operator=">=", threshold=0.8) is False


def test_less_than_or_equal_gate_passes_when_value_is_below_threshold():
    assert compare_metric(metric_value=0.15, operator="<=", threshold=0.2) is True


def test_less_than_or_equal_gate_fails_when_value_is_above_threshold():
    assert compare_metric(metric_value=0.25, operator="<=", threshold=0.2) is False


def test_equal_gate_passes_for_same_value():
    assert compare_metric(metric_value=1.0, operator="==", threshold=1.0) is True


def test_unsupported_operator_raises_error():
    with pytest.raises(ValueError):
        compare_metric(metric_value=0.8, operator="!=", threshold=0.8)