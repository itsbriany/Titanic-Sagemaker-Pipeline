from titanic_pipeline.config import (
    DEFAULT_ROC_AUC_THRESHOLD,
    get_metric_thresholds,
    get_raw_data_uri,
)


def test_get_raw_data_uri_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv("TITANIC_PIPELINE_RAW_DATA_URI", "s3://custom/raw.zip")

    assert get_raw_data_uri("ignored-bucket") == "s3://custom/raw.zip"


def test_get_raw_data_uri_builds_from_bucket(monkeypatch):
    monkeypatch.delenv("TITANIC_PIPELINE_RAW_DATA_URI", raising=False)

    assert get_raw_data_uri("demo-bucket") == "s3://demo-bucket/datasets/titanic/raw/titanic.zip"


def test_get_metric_thresholds_reads_environment(monkeypatch):
    monkeypatch.setenv("TITANIC_PIPELINE_F1_THRESHOLD", "0.81")
    monkeypatch.delenv("TITANIC_PIPELINE_ROC_AUC_THRESHOLD", raising=False)

    thresholds = get_metric_thresholds()

    assert thresholds.f1 == 0.81
    assert thresholds.roc_auc == DEFAULT_ROC_AUC_THRESHOLD
