PYTHON ?= python

.PHONY: install test lint local-preprocess local-train local-eval infra-synth

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

local-preprocess:
	$(PYTHON) preprocessing.py --input-dir data --train-dir /tmp/titanic-train --test-dir /tmp/titanic-test --meta-dir /tmp/titanic-meta

local-train:
	$(PYTHON) train.py --input-dir /tmp/titanic-train --meta-dir /tmp/titanic-meta --model-dir /tmp/titanic-model --output-dir /tmp/titanic-output

local-eval:
	$(PYTHON) evaluate.py --model-dir /tmp/titanic-model --test-dir /tmp/titanic-test --evaluation-dir /tmp/titanic-eval

infra-synth:
	cd infra && cdk synth
