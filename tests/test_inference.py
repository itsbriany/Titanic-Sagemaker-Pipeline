import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import input_fn, model_fn, output_fn, predict_fn
from titanic_pipeline.artifacts import save_model_artifacts
from titanic_pipeline.preprocessing import preprocess_training_dataframe

RAW_TITANIC_ROWS = [
    {
        "PassengerId": 1,
        "Survived": 0,
        "Pclass": 3,
        "Name": "Braund, Mr. Owen Harris",
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "A/5 21171",
        "Fare": 7.25,
        "Cabin": None,
        "Embarked": "S",
    },
    {
        "PassengerId": 2,
        "Survived": 1,
        "Pclass": 1,
        "Name": "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
        "Sex": "female",
        "Age": 38.0,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "PC 17599",
        "Fare": 71.2833,
        "Cabin": "C85",
        "Embarked": "C",
    },
    {
        "PassengerId": 3,
        "Survived": 1,
        "Pclass": 3,
        "Name": "Heikkinen, Miss. Laina",
        "Sex": "female",
        "Age": 26.0,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "STON/O2. 3101282",
        "Fare": 7.925,
        "Cabin": None,
        "Embarked": "S",
    },
]
@pytest.fixture
def trained_artifacts(tmp_path):
    processed_df, preprocessing_metadata = preprocess_training_dataframe(pd.DataFrame(RAW_TITANIC_ROWS))
    feature_df = processed_df.drop(columns=["Survived"])
    target = processed_df["Survived"]
    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
    model.fit(feature_df, target)

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    save_model_artifacts(
        model,
        str(model_dir),
        {
            "feature_columns": feature_df.columns.tolist(),
            "preprocessing": preprocessing_metadata,
        },
    )

    return {
        "model_dir": model_dir,
        "feature_df": feature_df,
        "feature_columns": feature_df.columns.tolist(),
        "raw_df": pd.DataFrame(RAW_TITANIC_ROWS).drop(columns=["Survived"]),
    }


def assert_valid_prediction(prediction, expected_len):
    assert set(prediction.keys()) == {"predictions", "probabilities"}
    assert len(prediction["predictions"]) == expected_len
    assert len(prediction["probabilities"]) == expected_len
    assert all(value in (0, 1) for value in prediction["predictions"])
    assert all(0.0 <= value <= 1.0 for value in prediction["probabilities"])


def test_model_fn_loads_expected_artifacts(trained_artifacts):
    artifacts = model_fn(str(trained_artifacts["model_dir"]))

    assert hasattr(artifacts["model"], "predict")
    assert artifacts["metadata"]["feature_columns"] == trained_artifacts["feature_columns"]


def test_input_fn_accepts_json_payload(trained_artifacts):
    payload = trained_artifacts["feature_df"].iloc[[0]].to_json(orient="records")

    parsed = input_fn(payload, "application/json")

    assert isinstance(parsed, pd.DataFrame)
    assert parsed.to_dict(orient="records") == trained_artifacts["feature_df"].iloc[[0]].to_dict(
        orient="records"
    )


def test_input_fn_accepts_csv_payload_and_bytes(trained_artifacts):
    payload = trained_artifacts["feature_df"].iloc[[0, 1]].to_csv(index=False, header=False).encode(
        "utf-8"
    )

    parsed = input_fn(payload, "text/csv")

    assert isinstance(parsed, pd.DataFrame)
    assert parsed.shape == (2, len(trained_artifacts["feature_columns"]))


def test_predict_fn_aligns_missing_columns(trained_artifacts):
    artifacts = model_fn(str(trained_artifacts["model_dir"]))
    partial = trained_artifacts["feature_df"].iloc[[0]].drop(columns=trained_artifacts["feature_columns"][-1])

    prediction = predict_fn(partial, artifacts)

    assert_valid_prediction(prediction, expected_len=1)


def test_predict_fn_accepts_raw_json_rows(trained_artifacts):
    artifacts = model_fn(str(trained_artifacts["model_dir"]))

    prediction = predict_fn(trained_artifacts["raw_df"].iloc[[0, 1]], artifacts)

    assert_valid_prediction(prediction, expected_len=2)


def test_full_inference_hook_chain_json(trained_artifacts):
    artifacts = model_fn(str(trained_artifacts["model_dir"]))
    payload = trained_artifacts["feature_df"].iloc[[0, 1]].to_json(orient="records")

    parsed = input_fn(payload, "application/json")
    prediction = predict_fn(parsed, artifacts)
    body, content_type = output_fn(prediction, None)

    decoded = json.loads(body)
    assert content_type == "application/json"
    assert_valid_prediction(decoded, expected_len=2)


def test_input_fn_rejects_unsupported_content_type():
    with pytest.raises(ValueError, match="Unsupported content type"):
        input_fn("{}", "application/octet-stream")


def test_output_fn_rejects_unsupported_accept():
    with pytest.raises(ValueError, match="Unsupported accept type"):
        output_fn({"predictions": [0], "probabilities": [0.1]}, "text/plain")


def test_predict_fn_rejects_missing_required_raw_columns(trained_artifacts):
    artifacts = model_fn(str(trained_artifacts["model_dir"]))
    malformed = trained_artifacts["raw_df"].iloc[[0]].drop(columns=["Pclass"])

    with pytest.raises(ValueError, match="Missing required raw input columns"):
        predict_fn(malformed, artifacts)
