import io
import json
import os

import joblib
import pandas as pd


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "feature_columns.json"), "r", encoding="utf-8") as handle:
        feature_columns = json.load(handle)
    return {
        "model": model,
        "feature_columns": feature_columns,
    }


def input_fn(request_body, request_content_type):
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")

    if request_content_type == "text/csv":
        return pd.read_csv(io.StringIO(request_body), header=None)

    if request_content_type == "application/json":
        payload = json.loads(request_body)
        if isinstance(payload, dict):
            return pd.DataFrame([payload])
        return pd.DataFrame(payload)

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_artifacts):
    features = model_artifacts["feature_columns"]
    model = model_artifacts["model"]

    if list(input_data.columns) != features:
        input_data.columns = features[: len(input_data.columns)]
        input_data = input_data.reindex(columns=features, fill_value=0)

    probabilities = model.predict_proba(input_data)[:, 1]
    predictions = model.predict(input_data)

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
