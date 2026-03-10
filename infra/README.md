# Titanic Pipeline Infrastructure

This CDK app creates the deployment Lambda and its dedicated execution role for the SageMaker pipeline.

Inside a SageMaker container, `AWS_REGION` is often already available, but `AWS_ACCOUNT_ID` usually is not. The simplest pattern is to export it from the active AWS identity first.

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

After deploy, export the stack output and run the pipeline with it:

```bash
export DEPLOYMENT_LAMBDA_ARN=<stack output DeploymentLambdaArn>
python pipeline.py
```
