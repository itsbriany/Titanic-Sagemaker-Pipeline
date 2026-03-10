# Titanic Pipeline Infrastructure

This CDK app creates the deployment Lambda, its dedicated execution role, and an SSM parameter that stores the Lambda ARN for the SageMaker pipeline.

Inside a SageMaker container, `AWS_REGION` is often already available, but `AWS_ACCOUNT_ID` usually is not. The simplest pattern is to export it from the active AWS identity first.

The Lambda code is packaged from `infra/lambda/`. This avoids recursive asset staging through `infra/cdk.out/`, which can otherwise produce repeated nested paths during CDK synthesis.

## Install AWS CDK

The AWS CDK CLI is a Node.js package. If it is not already installed in your SageMaker container, install it with:

```bash
npm install -g aws-cdk
cdk --version
```

## Deploy

```bash
cd infra
conda activate sagemaker-dev
pip install -r requirements.txt
export AWS_ACCOUNT_ID=$(python -c "import boto3; print(boto3.client('sts').get_caller_identity()['Account'])")
export AWS_REGION=${AWS_REGION:-$(python -c "import boto3; print(boto3.Session().region_name or 'us-east-1')")}
cdk bootstrap aws://$AWS_ACCOUNT_ID/$AWS_REGION
cdk deploy
```

If you hit a stale asset-packaging issue from an earlier run, remove `infra/cdk.out/` and retry the CDK commands.

After deploy, the Lambda ARN is stored in SSM Parameter Store at `/titanic-pipeline/deployment-lambda/arn`.

This is a good fit for CI/CD because the pipeline launcher can resolve the current Lambda ARN from a stable parameter name instead of hardcoding a stack output or account-specific ARN.

To run the pipeline from the SageMaker container:

```bash
export DEPLOYMENT_LAMBDA_SSM_PARAMETER=/titanic-pipeline/deployment-lambda/arn
python pipeline.py
```

If you prefer, you can still override that lookup explicitly:

```bash
export DEPLOYMENT_LAMBDA_ARN=<lambda arn>
python pipeline.py
```
