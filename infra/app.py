#!/usr/bin/env python3
import os

import aws_cdk as cdk

from titanic_pipeline_infra_stack import TitanicPipelineInfraStack


app = cdk.App()
account = (
    os.environ.get("AWS_ACCOUNT_ID")
    or os.environ.get("CDK_DEFAULT_ACCOUNT")
    or app.node.try_get_context("account")
)
region = (
    os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or os.environ.get("CDK_DEFAULT_REGION")
    or app.node.try_get_context("region")
    or "us-east-1"
)

TitanicPipelineInfraStack(
    app,
    "TitanicPipelineInfraStack",
    env=cdk.Environment(account=account, region=region),
)

app.synth()
