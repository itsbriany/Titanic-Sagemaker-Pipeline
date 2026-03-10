from pathlib import Path

from aws_cdk import CfnOutput, Duration, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_ssm as ssm
from constructs import Construct


class TitanicPipelineInfraStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        lambda_role = iam.Role(
            self,
            "TitanicDeploymentLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Execution role for the Titanic pipeline deployment Lambda.",
        )
        lambda_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AWSLambdaBasicExecutionRole"
            )
        )
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:CreateEndpointConfig",
                    "sagemaker:CreateEndpoint",
                    "sagemaker:UpdateEndpoint",
                    "sagemaker:DescribeEndpoint",
                ],
                resources=["*"],
            )
        )

        deploy_lambda = lambda_.Function(
            self,
            "TitanicDeploymentLambda",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="deploy.lambda_handler",
            code=lambda_.Code.from_asset(
                str(Path(__file__).resolve().parent / "lambda")
            ),
            role=lambda_role,
            timeout=Duration.seconds(60),
            description="Creates or updates the Titanic SageMaker endpoint from the pipeline.",
        )

        deployment_lambda_parameter = ssm.StringParameter(
            self,
            "DeploymentLambdaArnParameter",
            parameter_name="/titanic-pipeline/deployment-lambda/arn",
            string_value=deploy_lambda.function_arn,
            description="Deployment Lambda ARN for the Titanic SageMaker pipeline.",
        )

        CfnOutput(
            self,
            "DeploymentLambdaArn",
            value=deploy_lambda.function_arn,
            description="ARN for the deployment Lambda invoked by the SageMaker pipeline.",
            export_name="TitanicPipelineDeploymentLambdaArn",
        )
        CfnOutput(
            self,
            "DeploymentLambdaArnParameterName",
            value=deployment_lambda_parameter.parameter_name,
            description="SSM parameter storing the deployment Lambda ARN.",
            export_name="TitanicPipelineDeploymentLambdaArnParameterName",
        )
