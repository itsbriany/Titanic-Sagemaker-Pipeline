import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = None
    return metrics


def summarize_fold_metrics(fold_metrics):
    summary = {}
    keys = fold_metrics[0].keys()
    for key in keys:
        values = [m[key] for m in fold_metrics if m[key] is not None]
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        else:
            summary[key] = {"mean": None, "std": None}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/processing/model"))
    parser.add_argument("--output-dir", type=str, default="/opt/ml/output/data")
    parser.add_argument("--target-col", type=str, default="Survived")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    input_csv = os.path.join(args.input_dir, "train.csv")
    df = pd.read_csv(input_csv)

    y = df[args.target_col]
    X = df.drop(columns=[args.target_col])

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=args.random_state,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_pred, y_prob)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

        print(f"Fold {fold_idx}: {metrics}")

    summary = summarize_fold_metrics(fold_metrics)

    final_model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=args.random_state,
    )
    final_model.fit(X, y)

    model_path = os.path.join(args.model_dir, "model.joblib")
    columns_path = os.path.join(args.model_dir, "feature_columns.json")
    metrics_path = os.path.join(args.output_dir, "cross_validation_metrics.json")

    joblib.dump(final_model, model_path)
    with open(columns_path, "w", encoding="utf-8") as f:
        json.dump(X.columns.tolist(), f, indent=2)

    training_report = {
        "target_column": args.target_col,
        "num_rows": int(len(df)),
        "num_features": int(X.shape[1]),
        "n_splits": args.n_splits,
        "fold_metrics": fold_metrics,
        "summary_metrics": summary,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(training_report, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved feature columns to {columns_path}")
    print(f"Saved CV metrics to {metrics_path}")


if __name__ == "__main__":
    main()