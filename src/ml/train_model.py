"""Train and save the regression model.

Owner: Member 1 (feature/ml-tool).

Run:  python -m src.ml.train_model

Reads data/processed/, trains a scikit-learn regressor, and saves the model plus
preprocessing artifacts to models/ (e.g., models/model.joblib). Accuracy is not
graded, so keep the model simple (LinearRegression / RandomForestRegressor).
"""
from __future__ import annotations


import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.ml.prepare_data import (
    RANDOM_STATE, TARGET, ORDINAL_CATEGORIES,
    load_data, split_data, make_preprocessor,
)


def make_models() -> dict:
    """The three regressors compared in the notebook."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train(data_path: str | Path | None = None,
                   model_path: str | Path | None = None) -> dict:
    X, y = load_data(data_path)
    split = split_data(X, y)
    X_train, X_val, X_test = split["X_train"], split["X_val"], split["X_test"]
    y_train, y_val, y_test = split["y_train"], split["y_val"], split["y_test"]

    validation_rows = []
    fitted = {}
    for model_name, estimator in make_models().items():
        preprocessor, _ = make_preprocessor(X_train)
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_val)
        metrics = regression_metrics(y_val, predictions)
        validation_rows.append({"Model": model_name, **metrics})
        fitted[model_name] = pipeline

    validation_results = (
        pd.DataFrame(validation_rows).sort_values("RMSE").reset_index(drop=True)
    )
    best_name = str(validation_results.loc[0, "Model"])
    best_pipeline = fitted[best_name]

    test_predictions = best_pipeline.predict(X_test)
    test_metrics = regression_metrics(y_test, test_predictions)

    _, metadata = make_preprocessor(X_train)
    bundle = {
        "pipeline": best_pipeline,
        "input_columns": X.columns.tolist(),
        "target": TARGET,
        "ordinal_categories": ORDINAL_CATEGORIES,
        **metadata,
        "validation_best_model": best_name,
        "validation_results": validation_results.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "random_state": RANDOM_STATE,
    }

    destination = (
        Path(model_path) if model_path is not None
        else Path(__file__).resolve().parents[2] / "models" / "model.joblib"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)

    print("\nValidation results (sorted by RMSE):")
    print(validation_results.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print("\nHeld-out test metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    print(f"\nSaved model bundle to: {destination.resolve()}")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Student Performance Advisor ML model.")
    parser.add_argument("--data", type=Path, default=None, help="Optional CSV path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output .joblib path.")
    args = parser.parse_args()
    train(args.data, args.output)