PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install fmt lint typecheck test check seed run

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt

fmt:
	$(BIN)/black .

lint:
	$(BIN)/black --check .
	$(BIN)/flake8 .

typecheck:
	$(BIN)/mypy

test:
	$(BIN)/pytest --cov

check: lint typecheck test

seed:
	$(BIN)/python -m scripts.seed_artifacts --model-dir models

run:
	$(BIN)/gunicorn -c app/gunicorn.conf.py app.app:app
