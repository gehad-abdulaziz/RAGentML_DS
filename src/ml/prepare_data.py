
"""Data preparation for the ML regression tool.

Owner: Member 1 (feature/ml-tool).

Run: python -m src.ml.prepare_data

Pipeline: data/raw/<dataset>.csv -> clean + encode + split
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
TARGET = "Exam_Score"


ORDINAL_CATEGORIES = {
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Family_Income": ["Low", "Medium", "High"],
    "Teacher_Quality": ["Low", "Medium", "High"],
    "Peer_Influence": ["Negative", "Neutral", "Positive"],
    "Parental_Education_Level": [
        "High School", "College", "Postgraduate"
    ],
    "Distance_from_Home": ["Near", "Moderate", "Far"],
}


# Known nominal categorical columns.
NOMINAL_CATEGORIES = [
    "Extracurricular_Activities",
    "Internet_Access",
    "School_Type",
    "Learning_Disabilities",
    "Gender",
]


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def default_data_path() -> Path:
    """Return the default dataset path."""
    return (
        project_root()
        / "data"
        / "raw"
        / "StudentPerformanceFactors.csv"
    )


def identify_columns(
    X: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """Identify numeric, ordinal, and nominal columns."""

    ordinal_cols = [
        column
        for column in ORDINAL_CATEGORIES
        if column in X.columns
    ]

    nominal_cols = [
        column
        for column in NOMINAL_CATEGORIES
        if column in X.columns
        and column not in ordinal_cols
    ]

    # Detect any additional categorical columns.
    additional_nominal_cols = [
        column
        for column in X.columns
        if (
            pd.api.types.is_object_dtype(X[column])
            or isinstance(
                X[column].dtype,
                pd.CategoricalDtype
            )
            or pd.api.types.is_bool_dtype(X[column])
        )
        and column not in ordinal_cols
        and column not in nominal_cols
    ]

    nominal_cols.extend(additional_nominal_cols)

    numeric_cols = [
        column
        for column in X.columns
        if column not in ordinal_cols
        and column not in nominal_cols
    ]

    return numeric_cols, ordinal_cols, nominal_cols


def load_data(
    data_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV and separate features from target."""

    path = (
        Path(data_path)
        if data_path is not None
        else default_data_path()
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Put StudentPerformanceFactors.csv inside data/raw/."
        )

    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET}' was not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    df = df.dropna(
        subset=[TARGET]
    ).reset_index(drop=True)

    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].copy()

    numeric_cols, _, _ = identify_columns(X)

    for column in numeric_cols:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series
) -> dict[str, Any]:
    """Create 70% train / 15% validation / 15% test split."""

    X_train_full, X_test, y_train_full, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    val_fraction_of_remaining = (
        VALIDATION_SIZE / (1 - TEST_SIZE)
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_full,
            y_train_full,
            test_size=val_fraction_of_remaining,
            random_state=RANDOM_STATE,
        )
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def make_preprocessor(
    X: pd.DataFrame,
) -> tuple[ColumnTransformer, dict[str, list[str]]]:
    """Build the preprocessing pipeline."""

    numeric_cols, ordinal_cols, nominal_cols = (
        identify_columns(X)
    )

    transformers = []

    # Numeric features
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="median")
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )

    # Ordinal categorical features
    if ordinal_cols:
        ordinal_categories = [
            ORDINAL_CATEGORIES[column]
            for column in ordinal_cols
        ]

        transformers.append(
            (
                "ordinal",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            )
                        ),
                        (
                            "encoder",
                            OrdinalEncoder(
                                categories=ordinal_categories,
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            )
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ordinal_cols,
            )
        )

    # Nominal categorical features
    if nominal_cols:
        transformers.append(
            (
                "nominal",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                            strategy="most_frequent",
                            keep_empty_features=True
                            )   
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            )
                        ),
                    ]
                ),
                nominal_cols,
            )
        )

    if not transformers:
        raise ValueError(
            "No usable input features were detected."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    metadata = {
        "numeric_columns": numeric_cols,
        "ordinal_columns": ordinal_cols,
        "nominal_columns": nominal_cols,
    }

    return preprocessor, metadata


if __name__ == "__main__":

    X, y = load_data()

    numeric_cols, ordinal_cols, nominal_cols = (
        identify_columns(X)
    )

    print(
        f"Dataset loaded: {len(X)} rows, "
        f"{len(X.columns)} features"
    )

    print(f"Target: {TARGET}")

    print(
        f"Numeric columns ({len(numeric_cols)}): "
        f"{numeric_cols}"
    )

    print(
        f"Ordinal columns ({len(ordinal_cols)}): "
        f"{ordinal_cols}"
    )

    print(
        f"Nominal columns ({len(nominal_cols)}): "
        f"{nominal_cols}"
    )

    split = split_data(X, y)

    print("\nData split:")

    for name, values in split.items():
        print(f"{name}: {len(values)} rows")

    preprocessor, _ = make_preprocessor(
        split["X_train"]
    )

    print("\nPreprocessor created successfully.")
    print(preprocessor)