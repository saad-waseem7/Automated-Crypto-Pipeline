# Pinned to 3.11 to match the Docker image and CI. Testing on another minor
# version can pass locally and fail in the pipeline.
PYTHON ?= python3.11
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install fmt lint typecheck test check seed run docker-build smoke k8s-up k8s-down k8s-validate install-loadtest loadtest help

help: ## List targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

install: ## Create .venv (Python 3.11) and install dev dependencies
	@$(PYTHON) -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 'error: need Python 3.11 (got %d.%d). Install it or run: make install PYTHON=/path/to/python3.11' % sys.version_info[:2])"
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements/dev.txt

fmt: ## Format with black
	$(BIN)/black .

lint: ## black --check and flake8
	$(BIN)/black --check .
	$(BIN)/flake8 .

typecheck: ## mypy strict
	$(BIN)/mypy

test: ## pytest with coverage
	$(BIN)/pytest --cov

check: lint typecheck test ## lint, typecheck and test (what CI runs)

seed: ## Write a SYNTHETIC model into ./models for smoke tests
	$(BIN)/python -m scripts.seed_artifacts --model-dir models

run: ## Run the API with gunicorn on :5000
	$(BIN)/gunicorn -c app/gunicorn.conf.py app.app:app

docker-build: ## Build the API and seed images locally
	docker build -f app/Dockerfile --target runtime -t btc-forecast-api:local .
	docker build -f app/Dockerfile --target seed -t btc-forecast-seed:local .

smoke: docker-build ## Container smoke test
	scripts/smoke_test.sh btc-forecast-api:local btc-forecast-seed:local

k8s-up: ## Deploy to Minikube and run the first training job
	scripts/k8s_up.sh

k8s-down: ## Delete the btc-forecast namespace
	kubectl delete namespace btc-forecast --ignore-not-found

k8s-validate: ## Validate Kubernetes manifests offline
	$(BIN)/kubernetes-validate --strict -k 1.31 k8s/*.yaml

install-loadtest: ## Install Locust into .venv
	$(BIN)/pip install -r requirements/loadtest.txt

# Needs a running cluster (make k8s-up) and make install-loadtest.
loadtest: ## Run the 3 minute load test against Minikube
	$(BIN)/locust -f loadtest/locustfile.py --headless -u 200 -r 20 -t 3m \
		--host http://$$(minikube ip):30080 --csv loadtest/results
