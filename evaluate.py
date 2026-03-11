import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from titanic_pipeline.config import MetricThresholds
from titanic_pipeline.evaluation import evaluate_model
from titanic_pipeline.logging_utils import configure_logging


def main():
    configure_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Evaluate the Titanic model.")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--test-dir", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--evaluation-dir", type=str, default="/opt/ml/processing/evaluation")
    parser.add_argument("--target-col", type=str, default="Survived")
    parser.add_argument("--f1-threshold", type=float, default=0.7)
    parser.add_argument("--roc-auc-threshold", type=float, default=0.83)
    args = parser.parse_args()

    output_path = evaluate_model(
        model_dir=args.model_dir,
        test_dir=args.test_dir,
        evaluation_dir=args.evaluation_dir,
        target_col=args.target_col,
        thresholds=MetricThresholds(f1=args.f1_threshold, roc_auc=args.roc_auc_threshold),
    )
    logger.info("Saved evaluation report to %s", output_path)
    print(Path(output_path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
