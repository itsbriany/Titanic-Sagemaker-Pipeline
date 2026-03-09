import boto3
import json
import sagemaker

from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig


def get_pipeline(
    region="us-east-1",
    role=None,
    default_bucket="sagemaker-studio-970361933570-9nh5vyj1jvq",
    pipeline_name="TitanicPipeline",
):
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(
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

    deploy_processor = ScriptProcessor(
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
        base_job_name=f"{base_job_prefix}-deploy",
    )

    deploy_step = ProcessingStep(
        name="DeployTitanicModel",
        processor=deploy_processor,
        job_arguments=[
            "--region",
            region,
            "--role",
            role,
            "--model-data",
            step_train.properties.ModelArtifacts.S3ModelArtifacts,
            "--image-uri",
            sagemaker.image_uris.retrieve(
                framework="sklearn",
                region=region,
                version="1.2-1",
                image_scope="inference",
                instance_type=deployment_instance_type,
            ),
            "--endpoint-name",
            endpoint_name,
            "--instance-type",
            deployment_instance_type,
            "--execution-id",
            ExecutionVariables.PIPELINE_EXECUTION_ID,
        ],
        code="deploy.py",
        cache_config=cache_config,
    )

    deployment_rejected = FailStep(
        name="RejectModelDeployment",
        error_message=Join(
            on="",
            values=[
                "Deployment skipped because the evaluation thresholds were not met. ",
                "Required: f1 >= 0.7 and roc_auc >= 0.85. ",
                "Observed: f1=",
                f1_score,
                ", roc_auc=",
                roc_auc_score,
                ". Improve feature engineering, class balance handling, threshold tuning, or model hyperparameters, then rerun the pipeline.",
            ],
        ),
    )

    step_condition = ConditionStep(
        name="CheckEvaluationThresholds",
        conditions=[
            ConditionGreaterThanOrEqualTo(left=f1_score, right=0.7),
            ConditionGreaterThanOrEqualTo(left=roc_auc_score, right=0.85),
        ],
        if_steps=[deploy_step],
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
    pipeline = get_pipeline(region=region, role=role)
    validate_pipeline(pipeline)
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print(f"Started pipeline execution: {execution.arn}")
