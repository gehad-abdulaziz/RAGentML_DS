"""Data preparation for the ML regression tool.

Owner: Member 1 (feature/ml-tool).

Run:  python -m src.ml.prepare_data

Pipeline: data/raw/<dataset>.csv -> clean + encode + split -> data/processed/
"""
from __future__ import annotations

# TODO (Member 1): choose and record the dataset, target and features in docs/data_sources.md.
# Suggested: "Student Performance Factors" (Kaggle), target = Exam_Score,
# features include Hours_Studied, Attendance, Sleep_Hours.

# TODO (Member 1): define the feature list, their allowed ranges and the target name here
# so train_model.py and ml_tool.py can import and share them.
FEATURE_COLUMNS: list[str] = []  # TODO
TARGET_COLUMN: str = ""  # TODO
FEATURE_RANGES: dict[str, tuple[float, float]] = {}  # TODO: {"Hours_Studied": (0, 80), ...}


def load_raw_data(path: str):
    """Load the raw CSV into a pandas DataFrame.

    TODO: read from data/raw/, return DataFrame.
    """
    raise NotImplementedError


def clean_and_encode(df):
    """Handle missing values, encode categoricals, scale numerics if needed.

    TODO: save any fitted preprocessing objects (encoders/scalers) to models/ with joblib.
    """
    raise NotImplementedError


def split_and_save(df) -> None:
    """Train/test split and write results to data/processed/.

    TODO: save train.csv and test.csv.
    """
    raise NotImplementedError


def main() -> None:
    """TODO: orchestrate load_raw_data -> clean_and_encode -> split_and_save."""
    raise NotImplementedError


if __name__ == "__main__":
    main()