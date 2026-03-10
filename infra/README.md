# Titanic Pipeline Infrastructure

This CDK app creates the deployment Lambda and its dedicated execution role for the SageMaker pipeline.

## Deploy

```bash
cd infra
conda activate sagemaker-dev
pip install -r requirements.txt
cdk bootstrap aws://ACCOUNT_ID/us-east-1
cdk deploy -c account=ACCOUNT_ID -c region=us-east-1
```

After deploy, export the stack output and run the pipeline with it:

```bash
export DEPLOYMENT_LAMBDA_ARN=<stack output DeploymentLambdaArn>
python pipeline.py
```
