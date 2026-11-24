"""Tests for the ML tool. Owner: Member 1 (feature/ml-tool).

Run:  pytest tests/test_ml_tool.py -v
Save the output to logs/ (e.g., pytest ... | tee logs/ml_tool_tests.log).
"""
import pytest  # noqa: F401

from src.tools.ml_tool import predict_performance


def test_valid_input_returns_ok():
    """TODO: valid features -> status ok, prediction is a float, unit is a str, inputs_used is a dict."""
    pytest.skip("TODO (Member 1)")


def test_missing_feature_returns_error():
    """TODO: missing required feature -> status error with a clear message (no exception)."""
    pytest.skip("TODO (Member 1)")


def test_out_of_range_returns_error():
    """TODO: e.g. Hours_Studied = -5 or 500 -> status error."""
    pytest.skip("TODO (Member 1)")


def test_wrong_type_returns_error():
    """TODO: non-dict input or non-numeric value -> status error, never raises."""
    pytest.skip("TODO (Member 1)")


def test_more_study_hours_changes_prediction():
    """TODO: supports what-if use: two calls with different Hours_Studied give different predictions."""
    pytest.skip("TODO (Member 1)")