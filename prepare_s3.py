import logging
import os
import sys
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from titanic_pipeline.config import require_value
from titanic_pipeline.logging_utils import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)

BUCKET_NAME = require_value(
    "TITANIC_PIPELINE_BUCKET",
    os.environ.get("TITANIC_PIPELINE_BUCKET"),
    "Set TITANIC_PIPELINE_BUCKET before initializing S3 prefixes.",
)
FOLDERS = [
    "datasets/titanic/raw/",
    "datasets/titanic/extracted/",
    "datasets/titanic/features/",
    "pipelines/titanic/processing/",
    "pipelines/titanic/training/",
    "pipelines/titanic/evaluation/",
    "models/titanic/",
    "batch-inference/titanic/",
    "logs/titanic/",
]


def main():
    s3 = boto3.client("s3")
    for folder in FOLDERS:
        s3.put_object(Bucket=BUCKET_NAME, Key=f"{folder}.keep")
        LOGGER.info("Created %s", folder)
    LOGGER.info("S3 structure initialized in bucket %s", BUCKET_NAME)


if __name__ == "__main__":
    main()
