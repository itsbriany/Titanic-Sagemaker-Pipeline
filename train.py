import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from titanic_pipeline.logging_utils import configure_logging
from titanic_pipeline.training import train_model


def main():
    configure_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Train the Titanic model.")
    parser.add_argument("--input-dir", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/processing/model"))
    parser.add_argument("--output-dir", type=str, default="/opt/ml/output/data")
    parser.add_argument("--meta-dir", type=str, default="/opt/ml/input/data/meta")
    parser.add_argument("--target-col", type=str, default="Survived")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    outputs = train_model(
        input_dir=args.input_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        target_col=args.target_col,
        n_splits=args.n_splits,
        random_state=args.random_state,
        meta_dir=args.meta_dir,
    )
    logger.info("Saved model to %s", outputs["model_path"])
    logger.info("Saved metadata to %s", outputs["metadata_path"])
    logger.info("Saved CV metrics to %s", outputs["metrics_path"])


if __name__ == "__main__":
    main()
