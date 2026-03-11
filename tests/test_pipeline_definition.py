import json
from types import SimpleNamespace

import sagemaker
import sagemaker.estimator
import sagemaker.model
import sagemaker.processing
import sagemaker.s3

from titanic_pipeline.config import MetricThresholds
from titanic_pipeline.pipeline import get_pipeline


def test_pipeline_definition_exposes_threshold_parameters(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "test")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setattr(
        sagemaker.image_uris,
        "retrieve",
        lambda *args, **kwargs: "123456789012.dkr.ecr/test:latest",
    )
    monkeypatch.setattr(
        sagemaker.model.Model,
        "_upload_code",
        lambda self, key_prefix, repack=False: setattr(self, "uploaded_code", None),
    )
    monkeypatch.setattr(
        sagemaker.processing.ScriptProcessor,
        "_upload_code",
        lambda self, code, kms_key=None: "s3://demo-bucket/pipeline-code.tar.gz",
    )
    monkeypatch.setattr(
        sagemaker.session.Session,
        "default_bucket",
        lambda self: "demo-bucket",
    )
    monkeypatch.setattr(
        sagemaker.estimator.Framework,
        "_stage_user_code_in_s3",
        lambda self: SimpleNamespace(
            script_name="train.py",
            s3_prefix="s3://demo-bucket/source/sourcedir.tar.gz",
        ),
    )
    monkeypatch.setattr(
        sagemaker.s3.S3Uploader,
        "upload_string_as_file_body",
        lambda *args, **kwargs: "s3://demo-bucket/source/runproc.sh",
    )

    pipeline = get_pipeline(
        region="us-east-1",
        role="arn:aws:iam::123456789012:role/test-role",
        default_bucket="demo-bucket",
        deployment_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
        thresholds=MetricThresholds(f1=0.72, roc_auc=0.85),
    )
    definition = json.loads(pipeline.definition())

    parameter_names = {parameter["Name"] for parameter in definition["Parameters"]}
    assert "F1Threshold" in parameter_names
    assert "RocAucThreshold" in parameter_names

    step_names = {step["Name"] for step in definition["Steps"]}
    assert "CheckEvaluationThresholds" in step_names
    assert "EvaluateTitanicModel" in step_names
