import argparse
import logging
import sys
from pathlib import Path

import sagemaker

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from titanic_pipeline.config import (
    get_default_bucket,
    get_metric_thresholds,
    get_region,
    require_value,
)
from titanic_pipeline.logging_utils import configure_logging
from titanic_pipeline.pipeline import get_pipeline, resolve_deployment_lambda_arn, validate_pipeline


def main():
    configure_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Build and optionally execute the Titanic SageMaker pipeline.")
    parser.add_argument("--region", type=str, default=get_region())
    parser.add_argument("--default-bucket", type=str, default=get_default_bucket())
    parser.add_argument("--pipeline-name", type=str, default="TitanicPipeline")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    role = sagemaker.get_execution_role()
    default_bucket = require_value(
        "default_bucket",
        args.default_bucket,
        "Set TITANIC_PIPELINE_BUCKET or pass --default-bucket.",
    )
    thresholds = get_metric_thresholds()
    deployment_lambda_arn = resolve_deployment_lambda_arn(args.region)

    pipeline = get_pipeline(
        region=args.region,
        role=role,
        default_bucket=default_bucket,
        pipeline_name=args.pipeline_name,
        deployment_lambda_arn=deployment_lambda_arn,
        thresholds=thresholds,
    )
    validate_pipeline(pipeline)
    if args.validate_only:
        return

    pipeline.upsert(role_arn=role)
    start_parameters = {}
    if deployment_lambda_arn:
        start_parameters["DeploymentLambdaArn"] = deployment_lambda_arn
    execution = pipeline.start(parameters=start_parameters or None)
    logger.info("Started pipeline execution: %s", execution.arn)


if __name__ == "__main__":
    main()
