import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from titanic_pipeline.logging_utils import configure_logging
from titanic_pipeline.preprocessing import run_preprocessing


def main():
    configure_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Preprocess Titanic training data.")
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--train-dir", type=str, default="/opt/ml/processing/train")
    parser.add_argument("--test-dir", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--meta-dir", type=str, default="/opt/ml/processing/meta")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-col", type=str, default="Survived")
    args = parser.parse_args()

    outputs = run_preprocessing(
        input_dir=args.input_dir,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        meta_dir=args.meta_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        target_col=args.target_col,
    )
    logger.info("Wrote train data to %s", outputs["train_path"])
    logger.info("Wrote test data to %s", outputs["test_path"])
    logger.info("Wrote metadata to %s", outputs["meta_path"])


if __name__ == "__main__":
    main()
