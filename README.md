# Automated Crypto Pipeline

Production MLOps pipeline for 7-day BTC-USD close forecasting:
Flask API, Docker, GitHub Actions, Airflow, Kubeflow, Kubernetes.

Work is delivered part by part, one branch per part, reviewed and merged by the teammate.
See [docs/ASSIGNMENT_REVIEW.md](docs/ASSIGNMENT_REVIEW.md) for the spec critique, design decisions and roadmap.

## Status

| Part | Scope | State |
|------|-------|-------|
| A | Flask API, gunicorn, pytest | In review |
| B | Docker, Compose, GitHub Actions | Next |
| C | Airflow, ETL, training, Kubeflow | Planned |
| D | Kubernetes | Planned |
| E | Verification and report | Planned |

## Quick start (Ubuntu, Python 3.11 or newer)

```bash
make install     # venv plus dev dependencies
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
```
