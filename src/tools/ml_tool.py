"""ML regression tool.

Owner: Member 1 (feature/ml-tool).

Contract (see docs/contracts.md):
    success: {"status": "ok", "data": {"prediction": float, "unit": str, "inputs_used": dict}}
    failure: {"status": "error", "error": "<clear message>"}
The function must NEVER raise to the agent. It must also work on its own (no agent needed).
"""
from __future__ import annotations


def predict_performance(features: dict) -> dict:
    """Predict a student's academic performance from the given features.

    Args:
        features: e.g. {"Hours_Studied": 20, "Attendance": 90, "Sleep_Hours": 7}

    Returns:
        {"status": "ok", "data": {"prediction": float, "unit": str, "inputs_used": dict}}
        or {"status": "error", "error": "<clear message>"}

    TODO:
      - wrap the whole body in try/except and return the error dict on any failure
      - validate that features is a dict, required features are present, types/ranges are valid
        (use FEATURE_RANGES from src/ml/prepare_data.py)
      - load the model + preprocessing artifacts from models/ (cache after first load)
      - build the input row in the training feature order, predict
      - return the prediction with its unit and the inputs actually used
    """
    raise NotImplementedError("TODO (Member 1)")


if __name__ == "__main__":
    # TODO (Member 1): example call for running the tool independently:
    #   python -m src.tools.ml_tool
    print(predict_performance({"Hours_Studied": 20, "Attendance": 90, "Sleep_Hours": 7}))