import json
import logging
import os
from typing import Optional

import boto3
import sagemaker
from sagemaker.processing import FrameworkProcessor
from sagemaker.inputs import TrainingInput
from sagemaker.lambda_helper import Lambda
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.lambda_step import LambdaStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep

from titanic_pipeline.config import (
    DEFAULT_DEPLOYMENT_INSTANCE_TYPE,
    DEFAULT_ENDPOINT_NAME,
    DEFAULT_EVALUATION_INSTANCE_TYPE,
    DEFAULT_EVALUATION_PREFIX,
    DEFAULT_KFOLD_SPLITS,
    DEFAULT_MODEL_PREFIX,
    DEFAULT_PIPELINE_NAME,
    DEFAULT_PROCESSING_INSTANCE_TYPE,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TRAINING_INSTANCE_TYPE,
    MetricThresholds,
    get_deployment_lambda_parameter,
    get_metric_thresholds,
    get_raw_data_uri,
)

LOGGER = logging.getLogger(__name__)


def resolve_deployment_lambda_arn(region: str) -> str:
    direct_arn = os.environ.get("DEPLOYMENT_LAMBDA_ARN")
    if direct_arn:
        return direct_arn

    parameter_name = get_deployment_lambda_parameter()
    try:
        ssm_client = boto3.client("ssm", region_name=region)
        response = ssm_client.get_parameter(Name=parameter_name)
        return response["Parameter"]["Value"]
    except Exception:
        LOGGER.warning("Unable to resolve deployment lambda ARN from SSM parameter %s", parameter_name)
        return ""


def get_pipeline(
    *,
    region: str,
    role: Optional[str],
    default_bucket: str,
    pipeline_name: str = DEFAULT_PIPELINE_NAME,
    deployment_lambda_arn: str = "",
    raw_data_uri: Optional[str] = None,
    thresholds: Optional[MetricThresholds] = None,
):
    LOGGER.info("Building SageMaker pipeline %s in region %s", pipeline_name, region)
    thresholds = thresholds or get_metric_thresholds()
    raw_data_uri = raw_data_uri or get_raw_data_uri(default_bucket)
    if not raw_data_uri:
        raise ValueError(
            "Missing raw data URI. Set TITANIC_PIPELINE_RAW_DATA_URI or TITANIC_PIPELINE_BUCKET before running the pipeline."
        )

    boto_session = boto3.Session(region_name=region)
    source_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sagemaker_session = PipelineSession(boto_session=boto_session, default_bucket=default_bucket)

    if role is None:
        role = sagemaker.get_execution_role(sagemaker_session)

    raw_data_uri_param = ParameterString(name="RawDataUri", default_value=raw_data_uri)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value=DEFAULT_PROCESSING_INSTANCE_TYPE,
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value=DEFAULT_TRAINING_INSTANCE_TYPE,
    )
    evaluation_instance_type = ParameterString(
        name="EvaluationInstanceType",
        default_value=DEFAULT_EVALUATION_INSTANCE_TYPE,
    )
    deployment_instance_type = ParameterString(
        name="DeploymentInstanceType",
        default_value=DEFAULT_DEPLOYMENT_INSTANCE_TYPE,
    )
    endpoint_name = ParameterString(name="EndpointName", default_value=DEFAULT_ENDPOINT_NAME)
    deployment_lambda_arn_param = ParameterString(
        name="DeploymentLambdaArn",
        default_value=deployment_lambda_arn,
    )
    kfold_splits = ParameterInteger(name="KFoldSplits", default_value=DEFAULT_KFOLD_SPLITS)
    f1_threshold = ParameterFloat(name="F1Threshold", default_value=thresholds.f1)
    roc_auc_threshold = ParameterFloat(name="RocAucThreshold", default_value=thresholds.roc_auc)

    cache_config = CacheConfig(enable_caching=True, expire_after="1d")
    processing_processor = FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version="1.2-1",
        instance_type=processing_instance_type,
        instance_count=1,
        role=role,
        sagemaker_session=sagemaker_session,
        base_job_name="titanic-preprocess",
    )
    process_step_args = processing_processor.run(
        code="preprocessing.py",
        source_dir=source_dir,
        inputs=[ProcessingInput(source=raw_data_uri_param, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination=f"s3://{default_bucket}/datasets/titanic/features/train",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/test",
                destination=f"s3://{default_bucket}/datasets/titanic/features/test",
            ),
            ProcessingOutput(
                output_name="meta",
                source="/opt/ml/processing/meta",
                destination=f"s3://{default_bucket}/datasets/titanic/features/meta",
            ),
        ],
    )
    step_process = ProcessingStep(
        name="PreprocessTitanicData",
        step_args=process_step_args,
        cache_config=cache_config,
    )

    sklearn_estimator = SKLearn(
        entry_point="train.py",
        framework_version="1.2-1",
        source_dir=source_dir,
        role=role,
        instance_type=training_instance_type,
        instance_count=1,
        sagemaker_session=sagemaker_session,
        base_job_name="titanic-train",
        hyperparameters={
            "target-col": DEFAULT_TARGET_COLUMN,
            "n-splits": kfold_splits,
            "random-state": 42,
        },
        output_path=f"s3://{default_bucket}/{DEFAULT_MODEL_PREFIX}",
    )
    step_train = TrainingStep(
        name="TrainTitanicModel",
        estimator=sklearn_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "meta": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["meta"].S3Output.S3Uri,
                content_type="application/json",
            ),
        },
        cache_config=cache_config,
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    evaluation_processor = FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version="1.2-1",
        instance_type=evaluation_instance_type,
        instance_count=1,
        role=role,
        sagemaker_session=sagemaker_session,
        base_job_name="titanic-evaluate",
    )
    evaluate_step_args = evaluation_processor.run(
        code="evaluate.py",
        source_dir=source_dir,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{default_bucket}/{DEFAULT_EVALUATION_PREFIX}",
            ),
        ],
        arguments=[
            "--f1-threshold",
            Join(on="", values=[f1_threshold]),
            "--roc-auc-threshold",
            Join(on="", values=[roc_auc_threshold]),
        ],
    )
    step_evaluate = ProcessingStep(
        name="EvaluateTitanicModel",
        step_args=evaluate_step_args,
        property_files=[evaluation_report],
        cache_config=cache_config,
    )

    f1_score = JsonGet(
        step_name=step_evaluate.name,
        property_file=evaluation_report,
        json_path="binary_classification_metrics.f1",
    )
    roc_auc_score = JsonGet(
        step_name=step_evaluate.name,
        property_file=evaluation_report,
        json_path="binary_classification_metrics.roc_auc",
    )

    deploy_model = SKLearnModel(
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        entry_point="inference.py",
        framework_version="1.2-1",
        source_dir=source_dir,
        sagemaker_session=sagemaker_session,
    )
    step_create_model = ModelStep(
        name="CreateTitanicModel",
        step_args=deploy_model.create(instance_type=deployment_instance_type),
    )
    step_deploy = LambdaStep(
        name="DeployTitanicModel",
        lambda_func=Lambda(function_arn=deployment_lambda_arn_param, session=sagemaker_session),
        inputs={
            "region": region,
            "endpoint_name": endpoint_name,
            "instance_type": deployment_instance_type,
            "model_name": step_create_model.steps[0].properties.ModelName,
        },
        depends_on=[step_create_model],
    )
    deployment_rejected = FailStep(
        name="RejectModelDeployment",
        error_message=Join(
            on="",
            values=[
                "Deployment skipped because one or more evaluation thresholds were not met. Required: f1 >= ",
                f1_threshold,
                " and roc_auc >= ",
                roc_auc_threshold,
                ". Observed f1=",
                f1_score,
                ", roc_auc=",
                roc_auc_score,
                ".",
            ],
        ),
    )
    step_condition = ConditionStep(
        name="CheckEvaluationThresholds",
        conditions=[
            ConditionGreaterThanOrEqualTo(left=f1_score, right=f1_threshold),
            ConditionGreaterThanOrEqualTo(left=roc_auc_score, right=roc_auc_threshold),
        ],
        if_steps=[step_create_model, step_deploy],
        else_steps=[deployment_rejected],
    )

    return Pipeline(
        name=pipeline_name,
        parameters=[
            raw_data_uri_param,
            processing_instance_type,
            training_instance_type,
            evaluation_instance_type,
            deployment_instance_type,
            endpoint_name,
            deployment_lambda_arn_param,
            kfold_splits,
            f1_threshold,
            roc_auc_threshold,
        ],
        steps=[
            step_process,
            step_train,
            step_evaluate,
            step_condition,
        ],
        sagemaker_session=sagemaker_session,
    )


def validate_pipeline(pipeline) -> None:
    definition = json.loads(pipeline.definition())
    LOGGER.info("Pipeline has %s steps", len(definition["Steps"]))
    print(json.dumps(definition, indent=2))
