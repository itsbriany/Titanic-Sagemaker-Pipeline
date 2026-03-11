from titanic_pipeline.config import MetricThresholds
from titanic_pipeline.evaluation import build_evaluation_report


def test_build_evaluation_report_marks_model_eligible():
    report = build_evaluation_report(
        {"accuracy": 0.8, "precision": 0.7, "recall": 0.75, "f1": 0.8, "roc_auc": 0.9},
        MetricThresholds(f1=0.7, roc_auc=0.83),
    )

    assert report["deployment_eligible"] is True
    assert report["thresholds"] == {"f1": 0.7, "roc_auc": 0.83}


def test_build_evaluation_report_marks_model_ineligible():
    report = build_evaluation_report(
        {"accuracy": 0.8, "precision": 0.7, "recall": 0.75, "f1": 0.6, "roc_auc": 0.9},
        MetricThresholds(f1=0.7, roc_auc=0.83),
    )

    assert report["deployment_eligible"] is False
