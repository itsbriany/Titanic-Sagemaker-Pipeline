import json
import os
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from titanic_pipeline.config import DEFAULT_RANDOM_STATE, DEFAULT_TARGET_COLUMN

DROPPED_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin"]
SEX_MAPPING = {"male": 0, "female": 1}


def find_csv_in_dir(input_dir: str) -> str:
    csv_files = []
    for root, _, files in os.walk(input_dir):
        for file_name in files:
            if file_name.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file_name))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found under {input_dir}")
    return csv_files[0]


def load_dataframe(input_dir: str) -> pd.DataFrame:
    zip_files = []
    for root, _, files in os.walk(input_dir):
        for file_name in files:
            if file_name.lower().endswith(".zip"):
                zip_files.append(os.path.join(root, file_name))

    if zip_files:
        extract_dir = os.path.join(input_dir, "unzipped")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_files[0], "r") as archive:
            archive.extractall(extract_dir)
        csv_path = find_csv_in_dir(extract_dir)
    else:
        csv_path = find_csv_in_dir(input_dir)

    return pd.read_csv(csv_path)


def build_preprocessing_metadata(df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COLUMN) -> dict:
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' in dataset")

    working_df = df.copy()
    working_df = working_df.drop(columns=DROPPED_COLUMNS, errors="ignore")

    numeric_imputations = {}
    for column in ("Age", "Fare"):
        if column in working_df.columns:
            numeric_imputations[column] = float(working_df[column].median())

    categorical_imputations = {}
    if "Embarked" in working_df.columns:
        categorical_imputations["Embarked"] = str(working_df["Embarked"].mode(dropna=True)[0])

    raw_feature_columns = [column for column in working_df.columns if column != target_col]
    transformed_df = transform_raw_features(
        working_df.drop(columns=[target_col]),
        {
            "target_column": target_col,
            "dropped_columns": DROPPED_COLUMNS,
            "raw_feature_columns": raw_feature_columns,
            "numeric_imputations": numeric_imputations,
            "categorical_imputations": categorical_imputations,
            "sex_mapping": SEX_MAPPING,
            "categorical_columns": [],
            "categorical_levels": {},
        },
        allow_unseen_categories=True,
    )
    categorical_columns = transformed_df.attrs["categorical_columns"]
    categorical_levels = transformed_df.attrs["categorical_levels"]

    metadata = {
        "target_column": target_col,
        "dropped_columns": DROPPED_COLUMNS,
        "raw_feature_columns": raw_feature_columns,
        "numeric_imputations": numeric_imputations,
        "categorical_imputations": categorical_imputations,
        "sex_mapping": SEX_MAPPING,
        "categorical_columns": categorical_columns,
        "categorical_levels": categorical_levels,
        "feature_columns": transformed_df.columns.tolist(),
    }
    return metadata


def _validate_required_raw_columns(df: pd.DataFrame, metadata: dict) -> None:
    missing_columns = sorted(set(metadata["raw_feature_columns"]) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required raw input columns: {missing_columns}")


def transform_raw_features(
    df: pd.DataFrame,
    metadata: dict,
    *,
    allow_unseen_categories: bool = False,
) -> pd.DataFrame:
    working_df = df.copy()
    _validate_required_raw_columns(working_df, metadata)
    working_df = working_df[metadata["raw_feature_columns"]]

    for column, fill_value in metadata.get("numeric_imputations", {}).items():
        if column in working_df.columns:
            working_df[column] = working_df[column].fillna(fill_value)

    for column, fill_value in metadata.get("categorical_imputations", {}).items():
        if column in working_df.columns:
            working_df[column] = working_df[column].fillna(fill_value)

    sex_mapping = metadata.get("sex_mapping", {})
    if "Sex" in working_df.columns:
        unknown_values = sorted(
            {str(value) for value in working_df["Sex"].dropna().unique() if value not in sex_mapping}
        )
        if unknown_values:
            raise ValueError(f"Unsupported values for Sex: {unknown_values}")
        working_df["Sex"] = working_df["Sex"].map(sex_mapping).astype("int64")

    categorical_columns = []
    categorical_levels = {}
    for column in working_df.select_dtypes(include=["object", "category"]).columns.tolist():
        known_levels = metadata.get("categorical_levels", {}).get(column)
        current_values = working_df[column].dropna().astype(str)
        if known_levels:
            unseen_values = sorted(set(current_values) - set(known_levels))
            if unseen_values and not allow_unseen_categories:
                raise ValueError(f"Unsupported values for {column}: {unseen_values}")
            working_df[column] = pd.Categorical(working_df[column], categories=known_levels)
            categorical_columns.append(column)
            categorical_levels[column] = known_levels
        else:
            working_df[column] = working_df[column].astype(str)
            levels = sorted(current_values.unique().tolist())
            categorical_columns.append(column)
            categorical_levels[column] = levels
            working_df[column] = pd.Categorical(working_df[column], categories=levels)

    transformed_df = pd.get_dummies(working_df, columns=categorical_columns, drop_first=False)
    transformed_df.attrs["categorical_columns"] = categorical_columns
    transformed_df.attrs["categorical_levels"] = categorical_levels
    return transformed_df


def preprocess_training_dataframe(
    df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COLUMN
) -> Tuple[pd.DataFrame, dict]:
    metadata = build_preprocessing_metadata(df, target_col=target_col)
    feature_df = transform_raw_features(df.drop(columns=[target_col]), metadata)
    processed_df = pd.concat([df[target_col].reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    metadata["feature_columns"] = feature_df.columns.tolist()
    return processed_df, metadata


def run_preprocessing(
    input_dir: str,
    train_dir: str,
    test_dir: str,
    meta_dir: str,
    test_size: float,
    random_state: int = DEFAULT_RANDOM_STATE,
    target_col: str = DEFAULT_TARGET_COLUMN,
) -> Dict[str, str]:
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    Path(meta_dir).mkdir(parents=True, exist_ok=True)

    df = load_dataframe(input_dir)
    processed_df, preprocessing_metadata = preprocess_training_dataframe(df, target_col=target_col)

    train_df, test_df = train_test_split(
        processed_df,
        test_size=test_size,
        random_state=random_state,
        stratify=processed_df[target_col],
    )

    train_path = os.path.join(train_dir, "train.csv")
    test_path = os.path.join(test_dir, "test.csv")
    meta_path = os.path.join(meta_dir, "metadata.json")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    metadata = {
        "target_column": target_col,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "num_features": int(train_df.shape[1] - 1),
        "feature_columns": preprocessing_metadata["feature_columns"],
        "preprocessing": preprocessing_metadata,
    }
    Path(meta_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {"train_path": train_path, "test_path": test_path, "meta_path": meta_path}
