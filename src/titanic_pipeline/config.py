import os
from dataclasses import dataclass
from typing import Optional

import boto3

DEFAULT_REGION = "us-east-1"
DEFAULT_PIPELINE_NAME = "TitanicPipeline"
DEFAULT_ENDPOINT_NAME = "titanic-sklearn-endpoint"
DEFAULT_DEPLOYMENT_PARAMETER = "/titanic-pipeline/deployment-lambda/arn"
DEFAULT_DATA_PREFIX = "datasets/titanic"
DEFAULT_MODEL_PREFIX = "models/titanic"
DEFAULT_EVALUATION_PREFIX = "pipelines/titanic/evaluation"
DEFAULT_PROCESSING_INSTANCE_TYPE = "ml.m5.large"
DEFAULT_TRAINING_INSTANCE_TYPE = "ml.m5.large"
DEFAULT_EVALUATION_INSTANCE_TYPE = "ml.m5.large"
DEFAULT_DEPLOYMENT_INSTANCE_TYPE = "ml.m5.large"
DEFAULT_TARGET_COLUMN = "Survived"
DEFAULT_KFOLD_SPLITS = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_F1_THRESHOLD = 0.7
DEFAULT_ROC_AUC_THRESHOLD = 0.83


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value


def get_region(default: str = DEFAULT_REGION) -> str:
    return env("AWS_REGION") or env("AWS_DEFAULT_REGION") or boto3.Session().region_name or default


def get_default_bucket() -> Optional[str]:
    return env("TITANIC_PIPELINE_BUCKET")


def get_raw_data_uri(bucket: Optional[str] = None) -> Optional[str]:
    explicit_uri = env("TITANIC_PIPELINE_RAW_DATA_URI")
    if explicit_uri:
        return explicit_uri
    resolved_bucket = bucket or get_default_bucket()
    if not resolved_bucket:
        return None
    return f"s3://{resolved_bucket}/{DEFAULT_DATA_PREFIX}/raw/titanic.zip"


def get_deployment_lambda_parameter() -> str:
    return env("DEPLOYMENT_LAMBDA_SSM_PARAMETER", DEFAULT_DEPLOYMENT_PARAMETER) or DEFAULT_DEPLOYMENT_PARAMETER


def get_deployment_lambda_arn() -> Optional[str]:
    return env("DEPLOYMENT_LAMBDA_ARN")


def get_float_env(name: str, default: float) -> float:
    value = env(name)
    return float(value) if value is not None else default


@dataclass(frozen=True)
class MetricThresholds:
    f1: float = DEFAULT_F1_THRESHOLD
    roc_auc: float = DEFAULT_ROC_AUC_THRESHOLD


def get_metric_thresholds() -> MetricThresholds:
    return MetricThresholds(
        f1=get_float_env("TITANIC_PIPELINE_F1_THRESHOLD", DEFAULT_F1_THRESHOLD),
        roc_auc=get_float_env("TITANIC_PIPELINE_ROC_AUC_THRESHOLD", DEFAULT_ROC_AUC_THRESHOLD),
    )


def require_value(name: str, value: Optional[str], guidance: str) -> str:
    if value:
        return value
    raise ValueError(f"Missing required setting for {name}. {guidance}")
