# Pinned to 3.11 to match the Docker image and CI. Testing on another minor
# version can pass locally and fail in the pipeline.
PYTHON ?= python3.11
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install fmt lint typecheck test check seed run docker-build smoke k8s-up k8s-down k8s-validate loadtest

install:
	@$(PYTHON) -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 'error: need Python 3.11 (got %d.%d). Install it or run: make install PYTHON=/path/to/python3.11' % sys.version_info[:2])"
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

docker-build:
	docker build -f app/Dockerfile --target runtime -t btc-forecast-api:local .
	docker build -f app/Dockerfile --target seed -t btc-forecast-seed:local .

smoke: docker-build
	scripts/smoke_test.sh btc-forecast-api:local btc-forecast-seed:local

k8s-up:
	scripts/k8s_up.sh

k8s-down:
	kubectl delete namespace btc-forecast --ignore-not-found

k8s-validate:
	$(BIN)/kubernetes-validate --strict -k 1.31 k8s/*.yaml

# Needs: pip install -r requirements-loadtest.txt and a running cluster.
loadtest:
	locust -f loadtest/locustfile.py --headless -u 200 -r 20 -t 3m \
		--host http://$$(minikube ip):30080 --csv loadtest/results
