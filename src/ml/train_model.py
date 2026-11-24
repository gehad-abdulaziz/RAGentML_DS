"""Train and save the regression model.

Owner: Member 1 (feature/ml-tool).

Run:  python -m src.ml.train_model

Reads data/processed/, trains a scikit-learn regressor, and saves the model plus
preprocessing artifacts to models/ (e.g., models/model.joblib). Accuracy is not
graded, so keep the model simple (LinearRegression / RandomForestRegressor).
"""
from __future__ import annotations


def train() -> None:
    """Train the model on data/processed/train.csv.

    TODO:
      - load processed training data
      - fit a regressor
      - print simple metrics (e.g., R2, MAE) on the test split
      - joblib.dump the model to models/model.joblib
      - joblib.dump preprocessing artifacts (encoders/scalers, feature order) to models/
    """
    raise NotImplementedError


def main() -> None:
    """TODO: call train()."""
    raise NotImplementedError


if __name__ == "__main__":
    main()