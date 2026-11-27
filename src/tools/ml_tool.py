
"""ML regression tool.

Owner: Member 1 (feature/ml-tool).

Contract (see docs/contracts.md):
    success: {"status": "ok", "data": {"prediction": float, "unit": str, "inputs_used": dict}}
    failure: {"status": "error", "error": "<clear message>"}

The function must NEVER raise to the agent.
It must also work on its own (no agent needed).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from numbers import Real
import math

import joblib
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "machine learning models"
    / "all_features_model.joblib"
)


# Basic numeric input limits.
# These are validation limits, not values learned by the model.
NUMERIC_LIMITS = {
    "Hours_Studied": (0, 100),
}


@lru_cache(maxsize=1)
def load_model():
    """Load the saved All Features model bundle once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run the training notebook first."
        )

    return joblib.load(MODEL_PATH)


def _is_missing(value):
    """Check whether a value is None, NaN, or missing."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def predict_performance(features: dict) -> dict:
    """Predict a student's academic performance from the given features.

    Args:
        features: Dictionary containing the student's input features.

    Returns:
        Success:
        {
            "status": "ok",
            "data": {
                "prediction": float,
                "unit": "exam_score",
                "inputs_used": dict
            }
        }

        Failure:
        {
            "status": "error",
            "error": "<clear message>"
        }
    """

    try:
        # 1. Validate input type
        if not isinstance(features, dict):
            return {
                "status": "error",
                "error": "features must be a dictionary."
            }

        # 2. Load the saved model bundle
        bundle = load_model()

        required_features = bundle["input_columns"]

        # 3. Check for missing features
        missing_features = [
            feature
            for feature in required_features
            if feature not in features
            or _is_missing(features.get(feature))
        ]

        if missing_features:
            return {
                "status": "error",
                "error": (
                    f"Missing required feature(s): {missing_features}"
                )
            }

        # 4. Keep only required features in training order
        inputs_used = {
            feature: features[feature]
            for feature in required_features
        }

        # 5. Validate numeric feature types and values
        numeric_columns = bundle.get("numeric_columns", [])

        for column in numeric_columns:
            value = inputs_used[column]

            if isinstance(value, bool) or not isinstance(value, Real):
                return {
                    "status": "error",
                    "error": (
                        f"Feature '{column}' must be numeric. "
                        f"Received: {value!r}."
                    )
                }

            if not math.isfinite(float(value)):
                return {
                    "status": "error",
                    "error": (
                        f"Feature '{column}' must be a finite number."
                    )
                }

            # Validate configured numeric limits
            if column in NUMERIC_LIMITS:
                minimum, maximum = NUMERIC_LIMITS[column]

                if not minimum <= value <= maximum:
                    return {
                        "status": "error",
                        "error": (
                            f"Feature '{column}' must be between "
                            f"{minimum} and {maximum}."
                        )
                    }

        # 6. Validate categorical feature types
        categorical_columns = (
            bundle.get("ordinal_columns", [])
            + bundle.get("nominal_columns", [])
        )

        for column in categorical_columns:
            value = inputs_used[column]

            if not isinstance(value, str):
                return {
                    "status": "error",
                    "error": (
                        f"Feature '{column}' must be a string category. "
                        f"Received: {value!r}."
                    )
                }

        # 7. Build input DataFrame in original training order
        input_data = pd.DataFrame(
            [inputs_used],
            columns=required_features
        )

        # 8. Predict using the saved fitted Pipeline
        prediction = float(
            bundle["pipeline"].predict(input_data)[0]
        )

        # 9. Return the project contract
        return {
            "status": "ok",
            "data": {
                "prediction": prediction,
                "unit": "exam_score",
                "inputs_used": inputs_used
            }
        }

    except Exception as error:
        # The tool must never raise an exception to the Agent
        return {
            "status": "error",
            "error": (
                f"{type(error).__name__}: {str(error)}"
            )
        }


if __name__ == "__main__":
    # Run independently without requiring the Agent
    try:
        bundle = load_model()

        print("ML Tool loaded successfully!")
        print("\nRequired features:")

        for feature in bundle["input_columns"]:
            print("-", feature)

        print(
            "\nTo test prediction, provide values for all "
            "the required features."
        )

        # Incomplete example to demonstrate missing-feature validation
        example_features = {
            "Hours_Studied": 20,
            "Attendance": 90,
            "Sleep_Hours": 7
        }

        result = predict_performance(example_features)

        print("\nPrediction result:")
        print(result)

    except Exception as error:
        print(
            f"ML Tool error: {type(error).__name__}: {error}"
        )
        