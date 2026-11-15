# Automated Crypto Pipeline

Production MLOps pipeline for 7-day BTC-USD close forecasting:
Flask API, Docker, GitHub Actions, Airflow, Kubeflow, Kubernetes.

Work is delivered part by part, one branch per part, reviewed and merged by the teammate.
See [docs/ASSIGNMENT_REVIEW.md](docs/ASSIGNMENT_REVIEW.md) for the spec critique, design decisions and roadmap.

## Status

| Part | Scope | State |
|------|-------|-------|
| A | Flask API, gunicorn, pytest | In review |
| B | Docker, Compose, GitHub Actions | In review |
| C | Airflow, ETL, training, Kubeflow | In review |
| D | Kubernetes (manifests, HPA, ingress, load test, kind CI) | In review |
| E | Verification and report | Planned |

## Quick start (Ubuntu, Python 3.11 exactly, to match Docker and CI)

```bash
make install     # builds .venv with python3.11; use PYTHON=/path/to/python3.11 if it is not on PATH
make check       # black, flake8, mypy strict, pytest with coverage
make seed        # writes a SYNTHETIC model into ./models for local smoke tests
make run         # gunicorn on :5000
```

```bash
curl localhost:5000/health
curl localhost:5000/ready
curl -X POST localhost:5000/predict -H 'Content-Type: application/json' -d '{"forecast_days": 3}'
curl localhost:5000/metrics
```

## Run in Docker

```bash
cp .env.example .env                    # then change every value
docker compose run --rm seed            # write a SYNTHETIC model into the volume (first run only)
docker compose up -d --build            # API on http://localhost:5000
scripts/check_api.sh                    # black-box checks against the running API
docker compose --profile airflow up -d  # optional: Postgres and Airflow on :8080
make smoke                              # full container smoke test
```

## Run the pipeline

```bash
# Offline, with synthetic prices (what CI does):
python -m scripts.fake_binance --port 8765 &
export BINANCE_BASE_URLS=http://127.0.0.1:8765
python -m src.etl_fetch && python -m src.features && python -m src.train_forecast

# Real data (needs internet, no API key):
unset BINANCE_BASE_URLS
python -m src.etl_fetch && python -m src.features && python -m src.train_forecast
cat data/reports/last_train.json

# Full stack with Airflow (needs real secrets in .env, see .env.example):
docker compose --profile airflow up -d --build
docker compose exec airflow airflow dags trigger btc_etl_pipeline
```

Kubeflow: `pip install -r requirements-kfp.txt && python kubeflow/btc_kfp_pipeline.py` recompiles `kubeflow/btc_kfp_pipeline.yaml`.

## API

| Method and path | Purpose |
|-----------------|---------|
| `GET /health` | Liveness. Always 200 while the process is up. Reports uptime and model version. |
| `GET /ready` | Readiness. 503 until a model is loaded. |
| `POST /predict` | Body optional. `forecast_days` (1 to 7, default 7). Or `history` (at least 31 positive closes, oldest first) with `history_end_date` (YYYY-MM-DD). |
| `GET /metrics` | MAE, RMSE, MAPE and naive baseline from the latest validation run. |
| `POST /reload` | Optional accelerator. Requires `Authorization: Bearer $RELOAD_TOKEN`. Disabled (404) when the token is unset. |

Errors are always JSON: `{"error": {"code": "...", "message": "...", "details": [...]}}`.

## Configuration (environment)

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_DIR` | `models` | Artifact directory (`/app/models` in containers) |
| `MODEL_STAGE` | `production` | Label shown in `/health` |
| `MODEL_POLL_SECONDS` | `10` | How often each worker checks for a new model |
| `LOG_LEVEL` | `INFO` | Python log level |
| `PORT` | `5000` | Bind port |
| `WEB_CONCURRENCY` | `2` | gunicorn workers |
| `RELOAD_TOKEN` | unset | Enables `POST /reload` (16 characters minimum) |
| `MAX_HISTORY` | `5000` | Max values accepted in `history` |

## Layout

```
app/    Flask service
src/    numpy forecaster and artifact contract, shared by training and serving
tests/  pytest suite
scripts/seed_artifacts.py   synthetic model for smoke tests
k8s/    Kubernetes manifests (kustomize), see docs/ASSIGNMENT_REVIEW.md for the runbook
loadtest/   Locust file
```
