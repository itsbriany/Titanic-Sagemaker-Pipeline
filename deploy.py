import argparse
import re

import boto3
from botocore.exceptions import ClientError


def build_name(endpoint_name, execution_id, suffix):
    sanitized_endpoint = re.sub(r"[^a-zA-Z0-9-]", "-", endpoint_name.lower()).strip("-")
    sanitized_execution = re.sub(r"[^a-zA-Z0-9-]", "-", execution_id.lower()).strip("-")
    max_prefix_len = 63 - len(suffix) - 2
    combined_prefix = f"{sanitized_endpoint}-{sanitized_execution}".strip("-")
    trimmed_prefix = combined_prefix[:max_prefix_len].strip("-")
    return f"{trimmed_prefix}-{suffix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--model-data", type=str, required=True)
    parser.add_argument("--image-uri", type=str, required=True)
    parser.add_argument("--endpoint-name", type=str, default="titanic-sklearn-endpoint")
    parser.add_argument("--instance-type", type=str, default="ml.m5.large")
    parser.add_argument("--execution-id", type=str, required=True)
    args = parser.parse_args()

    boto_session = boto3.Session(region_name=args.region)
    sagemaker_client = boto_session.client("sagemaker")
    model_name = build_name(args.endpoint_name, args.execution_id, "model")
    endpoint_config_name = build_name(args.endpoint_name, args.execution_id, "config")

    sagemaker_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=args.role,
        PrimaryContainer={
            "Image": args.image_uri,
            "ModelDataUrl": args.model_data,
        },
    )

    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": args.instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    try:
        sagemaker_client.describe_endpoint(EndpointName=args.endpoint_name)
        sagemaker_client.update_endpoint(
            EndpointName=args.endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        action = "Updated"
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ValidationException":
            raise
        sagemaker_client.create_endpoint(
            EndpointName=args.endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        action = "Created"

    print(f"{action} endpoint: {args.endpoint_name}")
    print(f"Model name: {model_name}")
    print(f"Endpoint config name: {endpoint_config_name}")


if __name__ == "__main__":
    main()
