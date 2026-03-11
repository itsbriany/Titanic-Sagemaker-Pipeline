import json
from pathlib import Path
from typing import Dict

import joblib

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "model_metadata.json"
LEGACY_FEATURE_COLUMNS_FILENAME = "feature_columns.json"


def save_model_artifacts(model, model_dir: str, metadata: dict) -> Dict[str, str]:
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / MODEL_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {"model_path": str(model_path), "metadata_path": str(metadata_path)}


def load_model_artifacts(model_dir: str) -> dict:
    artifact_dir = Path(model_dir)
    model = joblib.load(artifact_dir / MODEL_FILENAME)
    metadata_path = artifact_dir / METADATA_FILENAME
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        feature_columns_path = artifact_dir / LEGACY_FEATURE_COLUMNS_FILENAME
        metadata = {"feature_columns": json.loads(feature_columns_path.read_text(encoding="utf-8"))}
    return {"model": model, "metadata": metadata}
