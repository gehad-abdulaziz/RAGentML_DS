"""Tests for the ML tool. Owner: Member 1 (feature/ml-tool).

Run:
    python -m pytest tests/test_ml_tool.py -v

Save the output to logs/:
    pytest tests/test_ml_tool.py -v | Tee-Object -FilePath logs/ml_tool_tests.log

These tests use the saved All Features bundle to build a valid input row
with the exact feature names and categorical labels used during training.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from src.tools.ml_tool import predict_performance


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "machine learning models"
    / "all_features_model.joblib"
)


@pytest.fixture(scope="session")
def model_bundle():
    """Load the saved All Features bundle."""
    if not MODEL_PATH.exists():
        pytest.fail(
            f"Model artifact not found: {MODEL_PATH}. "
            "Run the training notebook before running these tests."
        )

    return joblib.load(MODEL_PATH)


@pytest.fixture
def valid_features(model_bundle):
    """Build a valid input using the model's training metadata."""

    bundle = model_bundle
    features = {}

    # 1. Numeric feature examples
    numeric_examples = {
        "Hours_Studied": 20,
        "Attendance": 90,
        "Sleep_Hours": 7,
        "Previous_Scores": 75,
        "Tutoring_Sessions": 2,
        "Physical_Activity": 3,
    }

    for column in bundle["numeric_columns"]:
        features[column] = numeric_examples.get(column, 1)

    # 2. Ordinal categorical features
    ordinal_categories = bundle.get("ordinal_categories", {})

    for column in bundle["ordinal_columns"]:
        categories = ordinal_categories[column]

        if not categories:
            pytest.fail(
                f"No configured ordinal categories for {column}."
            )

        features[column] = categories[len(categories) // 2]

    # 3. Nominal categorical features
    # The notebook adds transformers conditionally.
    # Therefore, "nominal" may not exist if there are no nominal columns.

    nominal_columns = bundle.get("nominal_columns", [])

    if nominal_columns:

        pipeline = bundle["pipeline"]
        preprocessor = pipeline.named_steps["preprocessor"]

        # Get the transformers that actually exist
        # in the fitted ColumnTransformer.
        fitted_transformers = {
            name: transformer
            for name, transformer, _columns
            in preprocessor.transformers_
        }

        if "nominal" not in fitted_transformers:
            pytest.fail(
                "The bundle lists nominal columns, but the fitted "
                "preprocessor has no 'nominal' transformer. "
                f"Available transformers: {list(fitted_transformers)}"
            )

        nominal_pipeline = fitted_transformers["nominal"]
        nominal_encoder = nominal_pipeline.named_steps["encoder"]

        if len(nominal_encoder.categories_) != len(nominal_columns):
            pytest.fail(
                "The number of fitted nominal category lists does not "
                "match bundle['nominal_columns']."
            )

        for column, categories in zip(
            nominal_columns,
            nominal_encoder.categories_,
        ):

            if len(categories) == 0:
                pytest.fail(
                    f"No fitted nominal categories for {column}."
                )

            value = categories[0]

            # Convert numpy scalar values to Python values.
            features[column] = (
                value.item()
                if hasattr(value, "item")
                else value
            )

    # 4. Return features in the exact training order
    return {
        column: features[column]
        for column in bundle["input_columns"]
    }


def test_valid_input_returns_ok(valid_features):
    """Valid features should return a successful prediction."""

    result = predict_performance(valid_features)

    assert isinstance(result, dict)
    assert result.get("status") == "ok", result

    data = result.get("data")

    assert isinstance(data, dict)
    assert isinstance(data.get("prediction"), float)
    assert isinstance(data.get("unit"), str)
    assert isinstance(data.get("inputs_used"), dict)


def test_missing_feature_returns_error(valid_features):
    """Missing required features should return a clear error."""

    missing_feature = next(iter(valid_features))

    incomplete_features = valid_features.copy()
    incomplete_features.pop(missing_feature)

    result = predict_performance(incomplete_features)

    assert result["status"] == "error"
    assert missing_feature in result["error"]


def test_out_of_range_returns_error(valid_features):
    """An invalid Hours_Studied value should be rejected."""

    if "Hours_Studied" not in valid_features:
        pytest.skip(
            "Hours_Studied is not part of this model's input features."
        )

    invalid_features = valid_features.copy()
    invalid_features["Hours_Studied"] = -5

    result = predict_performance(invalid_features)

    assert result["status"] == "error"
    assert "Hours_Studied" in result["error"]


def test_wrong_type_returns_error(valid_features):
    """Wrong input types should return errors, not exceptions."""

    # Test a non-dictionary input
    non_dict_result = predict_performance(
        ["not", "a", "dictionary"]
    )

    assert non_dict_result["status"] == "error"

    # Test a wrong numeric feature type
    invalid_features = valid_features.copy()
    invalid_features["Hours_Studied"] = "twenty"

    wrong_type_result = predict_performance(invalid_features)

    assert wrong_type_result["status"] == "error"
    assert "Hours_Studied" in wrong_type_result["error"]


def test_more_study_hours_changes_prediction(valid_features):
    """Different study hours should support what-if predictions."""

    if "Hours_Studied" not in valid_features:
        pytest.skip(
            "Hours_Studied is not part of this model's input features."
        )

    low_hours = valid_features.copy()
    high_hours = valid_features.copy()

    low_hours["Hours_Studied"] = 1
    high_hours["Hours_Studied"] = 40

    low_result = predict_performance(low_hours)
    high_result = predict_performance(high_hours)

    assert low_result["status"] == "ok", low_result
    assert high_result["status"] == "ok", high_result

    assert low_result["data"]["prediction"] != pytest.approx(
        high_result["data"]["prediction"]
    )