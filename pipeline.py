import boto3
import json
import os
import sagemaker

from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.lambda_step import LambdaStep
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig


def resolve_deployment_lambda_arn(region):
    direct_arn = os.environ.get("DEPLOYMENT_LAMBDA_ARN")
    if direct_arn:
        return direct_arn

    parameter_name = os.environ.get(
        "DEPLOYMENT_LAMBDA_SSM_PARAMETER",
        "/titanic-pipeline/deployment-lambda/arn",
    )
    if not parameter_name:
        return ""

    try:
        ssm_client = boto3.client("ssm", region_name=region)
        response = ssm_client.get_parameter(Name=parameter_name)
        return response["Parameter"]["Value"]
    except Exception:
        return ""


def get_pipeline(
    region="us-east-1",
    role=None,
    deployment_lambda_arn=None,
    default_bucket="sagemaker-studio-970361933570-9nh5vyj1jvq",
    pipeline_name="TitanicPipeline",
):
    boto_session = boto3.Session(region_name=region)
    source_dir = os.path.dirname(os.path.abspath(__file__))
    sagemaker_session = PipelineSession(
        boto_session=boto_session,
        default_bucket=default_bucket,
    )

    if role is None:
        role = sagemaker.get_execution_role(sagemaker_session)

    raw_data_uri = ParameterString(
        name="RawDataUri",
        default_value=f"s3://{default_bucket}/datasets/titanic/raw/titanic.zip",
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.m5.large",
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.m5.large",
    )
    evaluation_instance_type = ParameterString(
        name="EvaluationInstanceType",
        default_value="ml.m5.large",
    )
    deployment_instance_type = ParameterString(
        name="DeploymentInstanceType",
        default_value="ml.m5.large",
    )
    endpoint_name = ParameterString(
        name="EndpointName",
        default_value="titanic-sklearn-endpoint",
    )
    deployment_lambda_arn = ParameterString(
        name="DeploymentLambdaArn",
        default_value=deployment_lambda_arn or "",
    )
    kfold_splits = ParameterInteger(
        name="KFoldSplits",
        default_value=5,
    )

    base_job_prefix = "titanic"

    # Cache configuration: enable caching with 1-day expiration
    cache_config = CacheConfig(
        enable_caching=True,
        expire_after="1d",
    )

    processing_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=region,
            version="1.2-1",
            instance_type=processing_instance_type,
        ),
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        role=role,
        sagemaker_session=sagemaker_session,
        base_job_name=f"{base_job_prefix}-preprocess",
    )

    step_process = ProcessingStep(
        name="PreprocessTitanicData",
        processor=processing_processor,
        inputs=[
            ProcessingInput(
                source=raw_data_uri,
                destination="/opt/ml/processing/input",
            ),
        ],
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
        code="preprocessing.py",
        cache_config=cache_config,
    )

    sklearn_estimator = SKLearn(
        entry_point="train.py",
        framework_version="1.2-1",
        role=role,
        instance_type=training_instance_type,
        instance_count=1,
        sagemaker_session=sagemaker_session,
        base_job_name=f"{base_job_prefix}-train",
        hyperparameters={
            "target-col": "Survived",
            "n-splits": kfold_splits,
            "random-state": 42,
        },
        output_path=f"s3://{default_bucket}/models/titanic",
    )

    step_train = TrainingStep(
        name="TrainTitanicModel",
        estimator=sklearn_estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            )
        },
        cache_config=cache_config,
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    evaluation_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=region,
            version="1.2-1",
            instance_type=evaluation_instance_type,
        ),
        command=["python3"],
        instance_type=evaluation_instance_type,
        instance_count=1,
        role=role,
        sagemaker_session=sagemaker_session,
        base_job_name=f"{base_job_prefix}-evaluate",
    )

    step_evaluate = ProcessingStep(
        name="EvaluateTitanicModel",
        processor=evaluation_processor,
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
                destination=f"s3://{default_bucket}/pipelines/titanic/evaluation",
            ),
        ],
        code="evaluate.py",
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

    deploy_lambda = Lambda(function_arn=deployment_lambda_arn, session=sagemaker_session)

    step_deploy = LambdaStep(
        name="DeployTitanicModel",
        lambda_func=deploy_lambda,
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
                "Deployment skipped because one or more evaluation thresholds were not met. ",
                "Required: f1 >= 0.7 and roc_auc >= 0.83. Observed f1=",
                f1_score,
                ", roc_auc=",
                roc_auc_score,
                ". Review the evaluation-step output and revisit feature engineering, threshold tuning, or model hyperparameters before rerunning the pipeline.",
            ],
        ),
    )

    step_condition = ConditionStep(
        name="CheckEvaluationThresholds",
        conditions=[
            ConditionGreaterThanOrEqualTo(left=f1_score, right=0.7),
            ConditionGreaterThanOrEqualTo(left=roc_auc_score, right=0.83),
        ],
        if_steps=[step_create_model, step_deploy],
        else_steps=[deployment_rejected],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            raw_data_uri,
            processing_instance_type,
            training_instance_type,
            evaluation_instance_type,
            deployment_instance_type,
            endpoint_name,
            deployment_lambda_arn,
            kfold_splits,
        ],
        steps=[
            step_process,
            step_train,
            step_evaluate,
            step_condition,
        ],
        sagemaker_session=sagemaker_session,
    )

    return pipeline


def validate_pipeline(pipeline):
    definition = json.loads(pipeline.definition())
    print("Pipeline has", len(definition["Steps"]), "steps")
    for step in definition["Steps"]:
        print("Step:", step["Name"], "Type:", step["Type"])
    print(json.dumps(definition, indent=4))

if __name__ == "__main__":
    region = boto3.Session().region_name or "us-east-1"
    role = sagemaker.get_execution_role()
    resolved_deployment_lambda_arn = resolve_deployment_lambda_arn(region)
    pipeline = get_pipeline(
        region=region,
        role=role,
        deployment_lambda_arn=resolved_deployment_lambda_arn,
    )
    validate_pipeline(pipeline)
    pipeline.upsert(role_arn=role)
    start_parameters = {}
    if resolved_deployment_lambda_arn:
        start_parameters["DeploymentLambdaArn"] = resolved_deployment_lambda_arn
    execution = pipeline.start(parameters=start_parameters or None)
    print(f"Started pipeline execution: {execution.arn}")
