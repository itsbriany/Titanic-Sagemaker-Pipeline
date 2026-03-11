import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from titanic_pipeline.artifacts import save_model_artifacts
from titanic_pipeline.config import DEFAULT_RANDOM_STATE, DEFAULT_TARGET_COLUMN
from titanic_pipeline.metrics import compute_classification_metrics, summarize_fold_metrics

LOGGER = logging.getLogger(__name__)


def _load_preprocessing_metadata(meta_dir: str) -> dict:
    metadata_path = Path(meta_dir) / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def train_model(
    input_dir: str,
    model_dir: str,
    output_dir: str,
    target_col: str = DEFAULT_TARGET_COLUMN,
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
    meta_dir: Optional[str] = None,
) -> Dict[str, str]:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    input_csv = os.path.join(input_dir, "train.csv")
    LOGGER.info("Loading training data from %s", input_csv)
    df = pd.read_csv(input_csv)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        y_pred = model.predict(X.iloc[val_idx])
        y_prob = model.predict_proba(X.iloc[val_idx])[:, 1]
        metrics = compute_classification_metrics(y.iloc[val_idx], y_pred, y_prob)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)
        LOGGER.info("Fold %s metrics: %s", fold_idx, metrics)

    final_model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state)
    final_model.fit(X, y)

    preprocessing_bundle = _load_preprocessing_metadata(meta_dir) if meta_dir else {}
    preprocessing_metadata = preprocessing_bundle.get("preprocessing", {})
    model_metadata = {
        "target_column": target_col,
        "feature_columns": X.columns.tolist(),
        "preprocessing": preprocessing_metadata,
        "training": {
            "num_rows": int(len(df)),
            "num_features": int(X.shape[1]),
            "n_splits": n_splits,
            "random_state": random_state,
        },
    }
    artifact_paths = save_model_artifacts(final_model, model_dir, model_metadata)

    report = {
        "target_column": target_col,
        "num_rows": int(len(df)),
        "num_features": int(X.shape[1]),
        "n_splits": n_splits,
        "fold_metrics": fold_metrics,
        "summary_metrics": summarize_fold_metrics(fold_metrics),
    }
    metrics_path = Path(output_dir) / "cross_validation_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Saved training report to %s", metrics_path)

    return {**artifact_paths, "metrics_path": str(metrics_path)}
