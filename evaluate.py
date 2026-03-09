import argparse
import json
import os
import tarfile

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def print_directory_tree(root="/opt/ml"):
    print(f"===== DIRECTORY TREE: {root} =====")
    for current_root, dirs, files in os.walk(root):
        level = current_root.replace(root, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(current_root)}/")
        for file_name in files:
            print(f"{indent}  {file_name}")


def print_sm_env():
    print("===== SAGEMAKER ENV =====")
    for key, value in sorted(os.environ.items()):
        if key.startswith("SM_"):
            print(f"{key}={value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--test-dir", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--evaluation-dir", type=str, default="/opt/ml/processing/evaluation")
    parser.add_argument("--target-col", type=str, default="Survived")
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()

    if args.debug:
        print_directory_tree()
        print_sm_env()

    os.makedirs(args.evaluation_dir, exist_ok=True)

    # If a SageMaker model artifact tarball is present, extract it in place first.
    model_tar_path = os.path.join(args.model_dir, "model.tar.gz")
    if os.path.exists(model_tar_path):
        with tarfile.open(model_tar_path, "r:gz") as tar:
            tar.extractall(path=args.model_dir)

    model_path = os.path.join(args.model_dir, "model.joblib")
    model = joblib.load(model_path)
    with open(os.path.join(args.model_dir, "feature_columns.json"), "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    test_df = pd.read_csv(os.path.join(args.test_dir, "test.csv"))
    y_true = test_df[args.target_col]
    X_test = test_df.drop(columns=[args.target_col])

    # Align columns defensively
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }

    report = {
        "regression": None,
        "binary_classification_metrics": metrics,
    }

    output_path = os.path.join(args.evaluation_dir, "evaluation.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved evaluation report to {output_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()