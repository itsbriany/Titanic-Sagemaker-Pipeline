import argparse
import json
import os
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def find_csv_in_dir(input_dir: str) -> str:
    csv_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found under {input_dir}")
    return csv_files[0]


def load_dataframe(input_dir: str) -> pd.DataFrame:
    zip_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".zip"):
                zip_files.append(os.path.join(root, f))

    if zip_files:
        extract_dir = os.path.join(input_dir, "unzipped")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_files[0], "r") as zf:
            zf.extractall(extract_dir)
        csv_path = find_csv_in_dir(extract_dir)
    else:
        csv_path = find_csv_in_dir(input_dir)

    return pd.read_csv(csv_path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

    if "Survived" not in df.columns:
        raise ValueError("Expected target column 'Survived' in Titanic dataset")

    # Basic imputations
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Binary encode Sex if present
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype("int64")

    # One-hot encode remaining categoricals except target
    feature_df = df.drop(columns=["Survived"])
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    feature_df = pd.get_dummies(feature_df, columns=categorical_cols, drop_first=False)

    processed = pd.concat([df["Survived"], feature_df], axis=1)
    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--train-dir", type=str, default="/opt/ml/processing/train")
    parser.add_argument("--test-dir", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--meta-dir", type=str, default="/opt/ml/processing/meta")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)
    os.makedirs(args.meta_dir, exist_ok=True)

    df = load_dataframe(args.input_dir)
    processed = preprocess(df)

    train_df, test_df = train_test_split(
        processed,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=processed["Survived"],
    )

    train_path = os.path.join(args.train_dir, "train.csv")
    test_path = os.path.join(args.test_dir, "test.csv")
    meta_path = os.path.join(args.meta_dir, "metadata.json")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    metadata = {
        "target_column": "Survived",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "num_features": int(train_df.shape[1] - 1),
        "columns": train_df.columns.tolist(),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote train data to {train_path}")
    print(f"Wrote test data to {test_path}")
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()