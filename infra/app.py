#!/usr/bin/env python3
import aws_cdk as cdk

from titanic_pipeline_infra_stack import TitanicPipelineInfraStack


app = cdk.App()

TitanicPipelineInfraStack(
    app,
    "TitanicPipelineInfraStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-east-1",
    ),
)

app.synth()
