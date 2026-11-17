# Architecture

A daily pipeline that fetches BTC-USDT prices, trains a 7-day close forecaster, gates it against a
naive baseline, and serves the promoted model over HTTP. It runs on a laptop with Docker Compose or
on Kubernetes.

## Components

```
                  Binance klines (public, no API key)
                              |
                              v
  Airflow DAG (Compose)   or   CronJob btc-train (Kubernetes)
  fetch -> features -> train -> gate -> promote
                              |
                              v
        model store: versions/<id>/ + CURRENT pointer   (Docker volume or PVC)
                              ^
                              | each worker polls CURRENT
                              |
        Flask API, gunicorn (2 workers x 2 threads per container)
        /health  /ready  /predict  /metrics  /reload
                              ^
        Service + NGINX Ingress (Kubernetes) or published port (Compose)
```

| Path | Role |
|------|------|
| `src/etl_fetch.py` | Paginated Binance fetch with retry, host fallback and an idempotent atomic CSV upsert. Only closed candles are stored. |
| `src/features.py` | Gap filling (at most 3 days), MA7, MA30, 30-day volatility, and the 7 training targets. |
| `src/forecast_core.py` | numpy ridge regression on lagged log-returns, one model per horizon. Shared by training and serving. |
| `src/train_forecast.py` | Backtest, gate, refit on all data, publish. Writes a JSON report. |
| `src/artifacts.py` | Versioned artifact store with an atomic `CURRENT` pointer. |
| `app/` | Flask service (`app.py`), settings (`config.py`), request validation (`schemas.py`), per-worker model holder (`model_store.py`). |
| `dags/btc_etl_pipeline.py` | Airflow DAG: four tasks, daily, no catch-up, retries with exponential backoff. |
| `kubeflow/` | The same three steps as a Kubeflow Pipelines v2 spec. Compiled and drift-checked in CI, not deployed. |
| `k8s/` | Kustomize manifests. See [OPERATIONS.md](OPERATIONS.md). |

## Model artifact contract

```
CURRENT                         name of the active version
versions/<id>/model.npz         coefficients (loaded with allow_pickle=False)
versions/<id>/metadata.json     schema, last date, last 31 closes
versions/<id>/metrics.json      MAE, RMSE, MAPE, naive baseline, gate decision
```

Version ids are timestamps with microseconds and must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`,
which blocks path traversal. `CURRENT` is replaced with a temp file and `os.replace`, so a reader
sees the old or the new version, never a partial one. There is no pickle anywhere.

## Design decisions

**The API and the trainer only share a file contract.** Neither imports the other's runtime. The API
image needs only numpy and Flask, and the model can be produced by Airflow, a Kubernetes Job or a
laptop.

**Models are pulled, not pushed.** A Kubernetes Service sends a call to one pod and one gunicorn
worker, so a "reload" request would leave the others stale. Every worker instead checks `CURRENT`
at most every `MODEL_POLL_SECONDS` (10 by default), and every replica converges without coordination.
While no model is loaded a worker checks on every request, so the first model is picked up
immediately. `POST /reload` exists only as a token-protected accelerator, and is disabled (404)
when no token is set. A model that fails to load is not retried until `CURRENT` changes, and the
previous model keeps serving.

**Liveness and readiness are different endpoints.** `/health` is always 200 while the process is up.
`/ready` returns 503 until a model is loaded. If both used one endpoint, an empty volume on first
deploy would either crash-loop the pod (liveness fails) or send traffic to a pod that cannot predict
(readiness passes). A fresh cluster therefore shows unready pods until the first training run
finishes, which is the intended behavior.

**Promotion is gated on a naive baseline, not an absolute threshold.** BTC daily closes are close
to a random walk, and the honest benchmark is "tomorrow equals today". The trainer backtests over
the last 60 forecast origins and promotes a candidate only if its MAPE is within 5% of the naive
baseline's. It does not compare against the incumbent model, because overlapping windows would
favor the incumbent, which already saw part of the data. A rejected candidate publishes nothing.

The model is not expected to beat the baseline, and on real data it does not: a typical run gives a
MAPE close to the naive one. The 5% tolerance is what lets a roughly equal model through, so the
gate protects against a regression, not a proof of skill. This is a pipeline demonstration, not a
trading signal. Do not use the forecasts for trading.

**Indicators are stored but not model inputs.** MA7, MA30 and volatility are computed and written
to `features.csv` for inspection. The ridge model uses lagged log-returns only, since moving
averages of past closes add no information beyond them.

**Only closed candles are used.** The open candle for the current UTC day would leak a partial
close into training, so ETL drops it.

**Airflow orchestrates, it does not own the code.** Tasks run our scripts in a separate numpy-only
virtualenv inside the Airflow image, so Airflow's pinned dependency set is never disturbed. Tasks
exchange data through files, not XCom.

**Deployment is hardened by default.** Containers run as uid 10001 with a read-only root
filesystem, no capabilities and no privilege escalation. The Kubernetes namespace enforces the
Pod Security `restricted` profile and pods do not mount a service account token.

## Known limits

- The Kubernetes volume is ReadWriteOnce. It works on a single node (Minikube, kind) and does not
  on a multi-node cluster, where the API pods and the training job could land on different nodes.
  Use a ReadWriteMany class or object storage there.
- Airflow runs under Compose and the training CronJob runs in the cluster, as two separate
  schedulers. Run one or the other against the same volume, not both.
- Kubeflow is a compiled spec only. A full Kubeflow install does not fit next to Minikube and
  Airflow on a 16 GB machine.
- `/metrics` returns the latest validation metrics as JSON. It is not a Prometheus endpoint.
- No network policies or rate limiting are configured. Add both before exposing this beyond a
  local cluster, along with TLS on the ingress and an authenticated `/metrics`.
- Secrets are plain Kubernetes Secrets, which are base64, not encrypted. Use an external secret
  manager for anything real.
