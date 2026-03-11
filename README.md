# Titanic SageMaker Pipeline

Production-style ML pipeline for the Kaggle Titanic dataset: reproducible preprocessing, train/evaluate/deploy separation, SageMaker workflow orchestration, infrastructure as code, and testable inference contracts.

## What this project demonstrates

- SageMaker pipeline orchestration for preprocessing, training, evaluation, model registration, and endpoint deployment
- A canonical model artifact contract that bundles the estimator with feature schema and preprocessing metadata
- AWS CDK infrastructure for the deployment Lambda used by the pipeline
- Local test coverage for inference behavior, evaluation/report contracts, pipeline construction, config resolution, and endpoint deployment logic

## Architecture

The repository keeps the original top-level entrypoints for SageMaker compatibility, but the implementation now lives in [`src/titanic_pipeline`](./src/titanic_pipeline):

- `preprocessing`: raw dataset loading, imputations, categorical handling, and schema capture
- `training`: cross-validated training and model artifact persistence
- `evaluation`: offline model evaluation and deployment-threshold reports
- `inference`: SageMaker-compatible `model_fn` / `input_fn` / `predict_fn` / `output_fn`
- `pipeline`: parameterized SageMaker pipeline definition
- `config`: shared environment-driven defaults and thresholds

## Model artifact contract

Each trained model package contains:

- `model.joblib`: serialized scikit-learn estimator
- `model_metadata.json`: target column, feature column order, training metadata, and preprocessing metadata required to transform raw passenger records safely at inference time

This keeps training, evaluation, and inference aligned on the same schema instead of duplicating preprocessing logic across scripts.

## Local workflow

Install dependencies and run checks:

```bash
make install
make test
make lint
```

To create a local VS Code debugger setup without committing personal settings:

```bash
cp .vscode/launch.example.json .vscode/launch.json
```

Run the local preprocessing/train/evaluate flow against the sample data already present in `data/`:

```bash
python preprocessing.py --input-dir data --train-dir /tmp/titanic-train --test-dir /tmp/titanic-test --meta-dir /tmp/titanic-meta
python train.py --input-dir /tmp/titanic-train --meta-dir /tmp/titanic-meta --model-dir /tmp/titanic-model --output-dir /tmp/titanic-output
python evaluate.py --model-dir /tmp/titanic-model --test-dir /tmp/titanic-test --evaluation-dir /tmp/titanic-eval
```

## AWS workflow

Required environment variables:

```bash
export TITANIC_PIPELINE_BUCKET=<s3 bucket for pipeline assets>
export DEPLOYMENT_LAMBDA_SSM_PARAMETER=/titanic-pipeline/deployment-lambda/arn
```

Infrastructure:

```bash
cd infra
pip install -r requirements.txt
cdk bootstrap aws://$AWS_ACCOUNT_ID/$AWS_REGION
cdk deploy
```

Pipeline execution:

```bash
python pipeline.py
```

To inspect the generated pipeline definition without starting execution:

```bash
python pipeline.py --validate-only
```

## Engineering decisions

- The baseline model remains logistic regression. The idea here is the surrounding production discipline, not model novelty.
- Inference accepts either engineered feature matrices or raw JSON passenger records. Raw requests are validated against the training-time schema and fail clearly on unsupported categories or missing required columns.
- Deployment thresholds are explicit configuration instead of hidden constants. The evaluation report records both the observed metrics and the gating thresholds.

## Repository notes

- `data/` is treated as local sample material for development and demo purposes, not as the source of truth for production artifacts.
- The deployed endpoint management logic lives in [`infra/lambda/deploy.py`](./infra/lambda/deploy.py).
