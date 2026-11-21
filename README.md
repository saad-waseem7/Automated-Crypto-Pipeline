# Automated Crypto Pipeline

An end-to-end MLOps pipeline that forecasts the next 7 daily closes of BTC-USDT. It fetches prices,
trains and gates a model, publishes it, and serves it over a Flask API that runs on Docker Compose
or Kubernetes with autoscaling.

The forecaster is a deliberately simple numpy ridge regression. On a near random walk like BTC it
performs about as well as "tomorrow equals today", which is the baseline it is gated against. The
value of the project is the pipeline around the model, not the forecast. Do not trade on it.

```
Binance klines -> Airflow / CronJob: fetch, features, train, gate -> model store -> Flask API
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Runbooks and troubleshooting:
[docs/OPERATIONS.md](docs/OPERATIONS.md).

## Quick start

Needs Ubuntu or similar, Docker, and Python 3.11 exactly (it matches the Docker image and CI).

```bash
make install     # builds .venv
make check       # black, flake8, mypy strict, pytest with coverage
make seed        # SYNTHETIC model into ./models, for local smoke tests only
make run         # gunicorn on http://127.0.0.1:5000
```

```bash
curl localhost:5000/health
curl localhost:5000/ready
curl -X POST localhost:5000/predict -H 'Content-Type: application/json' -d '{"forecast_days": 3}'
curl localhost:5000/metrics
```

A terminal client (standard library only) shows the same data as tables:

```bash
make cli-status     # liveness, readiness, served model version
make forecast       # 7-day forecast with a trend sparkline
make cli-metrics    # model versus naive baseline and the gate result
python -m scripts.btc_cli watch     # polls readiness and flags model changes
```

Point it at another API with `BTC_API_URL`, e.g. `make forecast BTC_API_URL=http://btc-forecast.local`.

Other ways to run it:

| Goal | Command |
|------|---------|
| Containers (API, Postgres, Airflow) | `cp .env.example .env`, then `make build`, then `docker compose run --rm seed && docker compose --profile airflow up -d` |
| Kubernetes on Minikube | `make k8s-up` (one time) |
| Day to day, no downloads | `make start`, `make stop`, `make status` |
| Real data, no Airflow | `python -m src.etl_fetch && python -m src.features && python -m src.train_forecast` |
| Load test | `make install-loadtest && make loadtest` |

Full instructions are in [docs/OPERATIONS.md](docs/OPERATIONS.md), including what to run after a reboot.

## API

| Method and path | Purpose |
|-----------------|---------|
| `GET /health` | Liveness. Always 200 while the process is up. Reports uptime and model version. |
| `GET /ready` | Readiness. 503 until a model is loaded. |
| `POST /predict` | Body optional. `forecast_days` (1 to 7, default 7). Or `history` (at least 31 positive closes, oldest first) with `history_end_date` (YYYY-MM-DD). |
| `GET /metrics` | MAE, RMSE, MAPE and naive baseline from the latest validation run. |
| `POST /reload` | Optional accelerator. Requires `Authorization: Bearer $RELOAD_TOKEN`. Disabled (404) when the token is unset. |

Errors are always JSON: `{"error": {"code": "...", "message": "...", "details": [...]}}`.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_DIR` | `models` | Artifact directory (`/app/models` in containers) |
| `MODEL_STAGE` | `production` | Label shown in `/health` |
| `MODEL_POLL_SECONDS` | `10` | How often each worker checks for a new model |
| `LOG_LEVEL` | `INFO` | Python log level |
| `PORT` | `5000` | Bind port |
| `WEB_CONCURRENCY` | `2` | gunicorn workers |
| `GUNICORN_THREADS` | `2` | threads per worker |
| `RELOAD_TOKEN` | unset | Enables `POST /reload` (16 characters minimum) |
| `MAX_HISTORY` | `5000` | Max values accepted in `history` |
| `MAX_BODY_BYTES` | `1048576` | Request body limit |
| `BINANCE_BASE_URLS` | Binance hosts | Comma-separated data hosts for the trainer, tried in order |

## Repository layout

```
app/            Flask service, settings, validation, Dockerfile
src/            ETL, features, forecaster, artifact store (shared by training and serving)
dags/           Airflow DAG
airflow/        Airflow image (adds an isolated training virtualenv)
kubeflow/       Kubeflow Pipelines spec (source and compiled YAML)
k8s/            Kubernetes manifests (kustomize)
loadtest/       Locust load test
scripts/        terminal client, deploy and smoke scripts, fake Binance server, synthetic seeder
requirements/   dev, kubeflow and load-test dependencies (runtime ones live in app/)
tests/          pytest suite
docs/           architecture and operations
.github/        CI/CD workflow and Dependabot
```

## Images and secrets

CI publishes `ghcr.io/saad-waseem7/automated-crypto-pipeline` on every push to `main`, tagged with the
full commit SHA and `latest`. Pull requests build and test but never push. CI authenticates with the
built-in `GITHUB_TOKEN`: short-lived, least privilege, nothing to rotate. No credentials are
committed. `.env` is git-ignored, and the Kubernetes reload token is generated at deploy time.
