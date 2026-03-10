import re
from pprint import pformat

import boto3
from botocore.exceptions import ClientError


def build_name(endpoint_name, unique_id, suffix):
    sanitized_endpoint = re.sub(r"[^a-zA-Z0-9-]", "-", endpoint_name.lower()).strip("-")
    sanitized_unique_id = re.sub(r"[^a-zA-Z0-9-]", "-", unique_id.lower()).strip("-")
    max_prefix_len = 63 - len(suffix) - 2
    combined_prefix = f"{sanitized_endpoint}-{sanitized_unique_id}".strip("-")
    trimmed_prefix = combined_prefix[:max_prefix_len].strip("-")
    return f"{trimmed_prefix}-{suffix}"


def lambda_handler(event, context):
    print("Received deployment event:")
    print(pformat(event))

    region = event["region"]
    endpoint_name = event["endpoint_name"]
    instance_type = event["instance_type"]
    model_name = event["model_name"]
    unique_id = context.aws_request_id if context is not None else "manual-invocation"

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    endpoint_config_name = build_name(endpoint_name, unique_id, "config")
    print(
        f"Preparing deployment for endpoint_name={endpoint_name}, "
        f"model_name={model_name}, endpoint_config_name={endpoint_config_name}, "
        f"instance_type={instance_type}, region={region}"
    )

    try:
        print(f"Creating endpoint config: {endpoint_config_name}")
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": instance_type,
                    "InitialVariantWeight": 1.0,
                }
            ],
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ValidationException":
            raise
        print(
            "Endpoint config creation returned ValidationException; "
            "continuing because the config may already exist."
        )

    try:
        print(f"Describing endpoint: {endpoint_name}")
        endpoint_description = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_status = endpoint_description.get("EndpointStatus")
        print(f"Endpoint exists with status: {endpoint_status}")

        if endpoint_status in {"Creating", "Updating", "SystemUpdating", "RollingBack"}:
            raise RuntimeError(
                f"Endpoint {endpoint_name} exists but is currently in status "
                f"{endpoint_status}; retry deployment after it reaches InService or Failed."
            )

        print(f"Updating endpoint: {endpoint_name}")
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        action = "Updated"
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        error_message = exc.response["Error"].get("Message", "")
        print(
            f"describe/update endpoint raised ClientError: code={error_code}, "
            f"message={error_message}"
        )
        if error_code != "ValidationException":
            raise
        print(f"Creating endpoint because it was not found: {endpoint_name}")
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        action = "Created"

    return {
        "status": action,
        "endpoint_name": endpoint_name,
        "endpoint_config_name": endpoint_config_name,
        "model_name": model_name,
    }
