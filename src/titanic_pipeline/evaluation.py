import json
import logging
import os
import tarfile
from pathlib import Path
from typing import Optional

import pandas as pd

from titanic_pipeline.artifacts import load_model_artifacts
from titanic_pipeline.config import DEFAULT_TARGET_COLUMN, MetricThresholds
from titanic_pipeline.metrics import compute_classification_metrics

LOGGER = logging.getLogger(__name__)


def ensure_extracted_model_artifacts(model_dir: str) -> None:
    model_tar_path = Path(model_dir) / "model.tar.gz"
    if model_tar_path.exists():
        LOGGER.info("Extracting model artifact archive from %s", model_tar_path)
        with tarfile.open(model_tar_path, "r:gz") as archive:
            archive.extractall(path=model_dir)


def build_evaluation_report(metrics: dict, thresholds: MetricThresholds) -> dict:
    return {
        "binary_classification_metrics": metrics,
        "thresholds": {"f1": thresholds.f1, "roc_auc": thresholds.roc_auc},
        "deployment_eligible": (
            metrics.get("f1") is not None
            and metrics.get("roc_auc") is not None
            and metrics["f1"] >= thresholds.f1
            and metrics["roc_auc"] >= thresholds.roc_auc
        ),
    }


def evaluate_model(
    model_dir: str,
    test_dir: str,
    evaluation_dir: str,
    target_col: str = DEFAULT_TARGET_COLUMN,
    thresholds: Optional[MetricThresholds] = None,
) -> str:
    thresholds = thresholds or MetricThresholds()
    Path(evaluation_dir).mkdir(parents=True, exist_ok=True)
    ensure_extracted_model_artifacts(model_dir)

    LOGGER.info("Loading model artifacts from %s", model_dir)
    artifacts = load_model_artifacts(model_dir)
    model = artifacts["model"]
    metadata = artifacts["metadata"]
    feature_columns = metadata["feature_columns"]

    test_df = pd.read_csv(os.path.join(test_dir, "test.csv"))
    y_true = test_df[target_col]
    X_test = test_df.drop(columns=[target_col]).reindex(columns=feature_columns, fill_value=0)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    report = build_evaluation_report(
        compute_classification_metrics(y_true, y_pred, y_prob),
        thresholds,
    )

    output_path = Path(evaluation_dir) / "evaluation.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Saved evaluation report to %s", output_path)
    return str(output_path)
