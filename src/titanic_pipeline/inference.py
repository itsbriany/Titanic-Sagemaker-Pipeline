import io
import json
from typing import List

import pandas as pd

from titanic_pipeline.artifacts import load_model_artifacts
from titanic_pipeline.preprocessing import transform_raw_features


def model_fn(model_dir):
    return load_model_artifacts(model_dir)


def input_fn(request_body, request_content_type):
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")

    if request_content_type == "text/csv":
        return pd.read_csv(io.StringIO(request_body), header=None)

    if request_content_type == "application/json":
        payload = json.loads(request_body)
        if isinstance(payload, dict):
            return pd.DataFrame([payload])
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        raise ValueError("JSON payload must be an object or a list of objects")

    raise ValueError(f"Unsupported content type: {request_content_type}")


def _looks_like_feature_matrix(input_data: pd.DataFrame, feature_columns: List[str]) -> bool:
    return list(input_data.columns) == feature_columns or (
        set(input_data.columns).issubset(set(feature_columns))
        and len(input_data.columns) <= len(feature_columns)
    )


def _to_feature_frame(input_data: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    feature_columns = metadata["feature_columns"]
    preprocessing_metadata = metadata.get("preprocessing") or {}

    if _looks_like_feature_matrix(input_data, feature_columns):
        aligned = input_data.copy()
        if all(isinstance(column, int) for column in aligned.columns):
            aligned.columns = feature_columns[: len(aligned.columns)]
        return aligned.reindex(columns=feature_columns, fill_value=0)

    if not preprocessing_metadata:
        raise ValueError("Model artifacts do not include preprocessing metadata for raw JSON inference")

    transformed = transform_raw_features(input_data, preprocessing_metadata)
    return transformed.reindex(columns=feature_columns, fill_value=0)


def predict_fn(input_data, model_artifacts):
    metadata = model_artifacts["metadata"]
    model = model_artifacts["model"]
    feature_frame = _to_feature_frame(input_data, metadata)

    probabilities = model.predict_proba(feature_frame)[:, 1]
    predictions = model.predict(feature_frame)
    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
    }


def output_fn(prediction, accept):
    if accept in (None, "*/*"):
        accept = "application/json"
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction), "application/json"
