import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

MODULE_PATH = Path(__file__).resolve().parents[1] / "infra" / "lambda" / "deploy.py"
SPEC = importlib.util.spec_from_file_location("deploy_lambda_module", MODULE_PATH)
DEPLOY_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DEPLOY_MODULE)
build_name = DEPLOY_MODULE.build_name
lambda_handler = DEPLOY_MODULE.lambda_handler


def _client_error(code: str, message: str):
    return ClientError({"Error": {"Code": code, "Message": message}}, "test")


class FakeSageMakerClient:
    def __init__(self, *, endpoint_exists=True, endpoint_status="InService"):
        self.endpoint_exists = endpoint_exists
        self.endpoint_status = endpoint_status
        self.created_configs = []
        self.updated = []
        self.created = []

    def create_endpoint_config(self, **kwargs):
        self.created_configs.append(kwargs)

    def describe_endpoint(self, EndpointName):
        if not self.endpoint_exists:
            raise _client_error("ValidationException", "not found")
        return {"EndpointName": EndpointName, "EndpointStatus": self.endpoint_status}

    def update_endpoint(self, **kwargs):
        self.updated.append(kwargs)

    def create_endpoint(self, **kwargs):
        self.created.append(kwargs)


def test_build_name_sanitizes_and_trims():
    name = build_name("Titanic Endpoint", "abc123", "config")

    assert name.endswith("-config")
    assert " " not in name
    assert len(name) <= 63


def test_lambda_handler_updates_existing_endpoint(monkeypatch):
    client = FakeSageMakerClient()

    class FakeSession:
        def client(self, name):
            assert name == "sagemaker"
            return client

    monkeypatch.setattr(DEPLOY_MODULE.boto3, "Session", lambda region_name: FakeSession())
    response = lambda_handler(
        {
            "region": "us-east-1",
            "endpoint_name": "demo-endpoint",
            "instance_type": "ml.m5.large",
            "model_name": "model-1",
        },
        SimpleNamespace(aws_request_id="request-1"),
    )

    assert response["status"] == "Updated"
    assert client.updated


def test_lambda_handler_creates_missing_endpoint(monkeypatch):
    client = FakeSageMakerClient(endpoint_exists=False)

    class FakeSession:
        def client(self, name):
            return client

    monkeypatch.setattr(DEPLOY_MODULE.boto3, "Session", lambda region_name: FakeSession())
    response = lambda_handler(
        {
            "region": "us-east-1",
            "endpoint_name": "demo-endpoint",
            "instance_type": "ml.m5.large",
            "model_name": "model-1",
        },
        SimpleNamespace(aws_request_id="request-1"),
    )

    assert response["status"] == "Created"
    assert client.created


def test_lambda_handler_blocks_when_endpoint_is_busy(monkeypatch):
    client = FakeSageMakerClient(endpoint_exists=True, endpoint_status="Updating")

    class FakeSession:
        def client(self, name):
            return client

    monkeypatch.setattr(DEPLOY_MODULE.boto3, "Session", lambda region_name: FakeSession())

    with pytest.raises(RuntimeError, match="currently in status Updating"):
        lambda_handler(
            {
                "region": "us-east-1",
                "endpoint_name": "demo-endpoint",
                "instance_type": "ml.m5.large",
                "model_name": "model-1",
            },
            SimpleNamespace(aws_request_id="request-1"),
        )
