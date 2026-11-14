# Assignment 3 Review and Execution Roadmap

Scope: critique of the assignment spec, the decisions we took because of it, and the plan for Parts B to E.
Part A is implemented on branch `part-a/flask-api`.

## 1. Architectural flaws in the spec

| # | Flaw | Why it breaks | Our decision |
|---|------|---------------|--------------|
| 1 | Circular dependency: Part A serves a model that Part C produces | The API cannot be demonstrated or tested in isolation | Define a file artifact contract first (`src/artifacts.py`). The API only knows the contract. Trainers write it later. |
| 2 | Three disconnected runtimes: Airflow in Docker Compose (B2), API in Kubernetes (D), model on a PVC (D2) | Compose Airflow cannot write to a PVC inside the cluster. The "reload" HTTP call from Compose to `btc-forecast.local` needs host DNS and routing. | Run the trainer as a Kubernetes Job on the same PVC (Part D), and let Airflow trigger it. Airflow in Compose stays a scheduler only. |
| 3 | "Notify the API to reload" (C1) | A Service load-balances to one pod and one gunicorn worker. The other 2+ replicas and workers stay stale. | Pull, not push. Every worker polls the tiny `CURRENT` file. `POST /reload` exists only as a token-protected accelerator. Verified live with 2 workers. |
| 4 | Liveness and readiness both use `/health` (D3) | With an empty PVC on first deploy there is no model. If `/health` fails, liveness kills the pod forever. If it passes, readiness lets traffic hit a 503. | `/health` is liveness and always 200. `/ready` returns 503 until a model loads. Point D3 readiness at `/ready` and justify it in the report. |
| 5 | HPA `minReplicas: 2` vs Deployment `replicas: 3` | HPA scales to 2 after the stabilization window and `kubectl apply` fights it | Keep 3 in the manifest as required and document the scale-down to 2 in the report. |
| 6 | `/metrics` | Reads as the Prometheus endpoint but the spec wants model metrics as JSON | JSON, as specified. Prometheus format is out of scope. |
| 7 | Promotion on "threshold criteria" with a 7-day test window (C1) | n=7 of a near random walk is noise. An absolute MAPE threshold either always passes or never does. | Walk-forward backtest over 60+ origins, scored against a naive baseline on the same origins. Promote only if not worse than the baseline and not worse than the incumbent. |
| 8 | Airflow, statsmodels, xgboost, prophet in one environment (section 3) | Airflow needs a constraints file. Combined resolution fails or downgrades things silently. Prophet pulls cmdstan, hundreds of MB. | Separate images. Airflow image has no ML libs. Training runs in its own image. Prophet is not used. |
| 9 | Full Kubeflow on a 16 GB, no GPU laptop | Kubeflow plus Minikube plus Airflow plus Docker does not fit | C2 only requires authoring and compiling to YAML. Compile, do not deploy. State this in the report. |
| 10 | `MODEL_STAGE` config, "register model artifacts" | Implies an MLflow registry that is not in scope | File registry: versioned dirs plus `CURRENT`. `MODEL_STAGE` is a label only. |
| 11 | CD pushes on "every push or pull request" (B3, B4) | Fork PRs have no secrets. Pushing images from PRs pollutes the registry. | PRs: lint, test, build only. Push only on `main`. |
| 12 | CoinGecko listed as a source | The free tier limits history to about 365 days. yfinance is an unofficial scraper that gets rate limited. Binance blocks some regions and US CI runners. | Primary: Binance klines (BTCUSDT, paginated). Fallback: yfinance. CI never touches the network. |
| 13 | Secrets as a Kubernetes `Secret` in git | Base64 is not encryption | Commit placeholders only. Real values via `kubectl create secret` or env. |

## 2. Unrealistic assumptions

- **Forecast value.** BTC daily close is close to a random walk. Expect to match, not beat, a naive baseline. Our synthetic smoke run shows 4.62 vs 4.69 percent MAPE, which is the realistic outcome. Grading rewards the pipeline, so do not chase accuracy. LSTM and Colab are unnecessary and add deployment risk.
- **Memory limit 512Mi.** pandas plus statsmodels plus scipy per gunicorn worker can exceed it. We avoided this: the API image needs only numpy and Flask.
- **gunicorn default workers.** The textbook `2 x cores + 1` reads host cores inside a container and OOMs against a 500m CPU limit. We default to 2 workers with 2 threads.
- **Incomplete daily candle.** Fetching "today" returns a partial candle. Training on it leaks a wrong close. ETL must drop the current UTC day.
- **Load test on one laptop.** The load generator and the cluster compete for the same CPU, so the HPA graph will be noisy. Run Locust with modest users and say so.
- **Ingress on Minikube/Kind.** Needs the addon (Minikube) or `extraPortMappings` at cluster creation (Kind), plus a `/etc/hosts` entry. Not "just apply the YAML".
- **metrics-server.** HPA does nothing without it. Minikube: `minikube addons enable metrics-server`. Kind: install it with `--kubelet-insecure-tls`.
- **Image visibility.** The cluster needs the image. Use `minikube image load` or `kind load docker-image` with `imagePullPolicy: IfNotPresent`. Pulling from a private registry fails silently into `ImagePullBackOff`.
- **RWO PVC with 3 replicas.** Fine on single-node Minikube. On multi-node Kind pods on other nodes cannot mount it. Use single-node clusters.

## 3. Part A: what was built

```
app/
  app.py            Flask factory; /health /ready /predict /metrics /reload; JSON errors
  config.py         Env settings, fail-fast validation
  schemas.py        Strict /predict validation (reports all problems at once)
  model_store.py    Per-worker hot-reloading model holder (poll, keep old on failure)
  gunicorn.conf.py  2 workers x 2 threads, /dev/shm heartbeat dir
  requirements.txt  Flask, gunicorn, numpy (runtime only)
src/
  forecast_core.py  numpy ridge, direct 7-horizon, shared by train and serve
  artifacts.py      Versioned artifact writer/reader, atomic CURRENT, path-traversal guard
scripts/seed_artifacts.py   Synthetic model for smoke tests (tagged data_source=synthetic)
tests/              120 tests, 99 percent coverage
```

Artifact contract (`MODEL_DIR`, mounted at `/app/models` in the container):

```
CURRENT                         active version name
versions/<ver>/model.npz        coefficients, loaded with allow_pickle=False
versions/<ver>/metadata.json    schema, last_date, last 31 closes
versions/<ver>/metrics.json     mae, rmse, mape, baseline_naive
```

Run command differs from the spec example: from the repo root use `gunicorn -c app/gunicorn.conf.py app.app:app`, because `app/` is a package. Mention this in the report.

## 4. Roadmap

### Part B: Docker, Compose, CI/CD (branch `part-b/containers-ci`)

Files:

```
app/Dockerfile
.dockerignore
docker-compose.yml
.github/workflows/ci_cd.yml
```

Build order: Dockerfile, run it, Compose, CI, CD.

Dockerfile pattern (build context is the repo root, `docker build -f app/Dockerfile .`):

- Stage 1 `builder`: `python:3.11-slim`, create `/opt/venv`, `pip install --no-cache-dir -r app/requirements.txt`. Copy requirements before source for layer caching.
- Stage 2 `runtime`: `python:3.11-slim`, copy `/opt/venv` only, copy `app/` and `src/` into `/app`, `useradd -u 10001`, `USER 10001`, `ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/app/models`, `WORKDIR /app`, `EXPOSE 5000`.
- `HEALTHCHECK` uses `python -c "urllib.request.urlopen(...)"` since slim has no curl.
- `CMD ["gunicorn","-c","app/gunicorn.conf.py","app.app:app"]` in exec form so SIGTERM reaches gunicorn.
- Python 3.11, not the 3.10 in the spec example: numpy 2.3 and 2.4 need 3.11 or newer. numpy is pinned to 2.4.6 because numpy 2.5 has no Python 3.11 wheels. Moving to a newer numpy means moving the Dockerfile, CI and Makefile to 3.12 together.

Compose:

- Services: `api`, `postgres`, `airflow-init`, `airflow` (LocalExecutor, official `apache/airflow` image pinned to a version, installed against its constraints file).
- Named volume `models` shared as `/app/models` on `api` (read-only) and the trainer (read-write).
- `depends_on` with `condition: service_healthy`. Postgres `healthcheck` with `pg_isready`.
- Airflow needs about 3 GB. Do not add CeleryExecutor, Redis or the triggerer.

CI workflow:

- Triggers: `push` to `main`, `pull_request`. `concurrency` cancels superseded runs. `permissions: contents: read`.
- Job `test`: Python 3.11, pip cache, `black --check`, `flake8`, `mypy`, `pytest --cov`.
- Job `image` (needs `test`): buildx with GHA cache. On PRs `push: false`. On `main`, log in with `DOCKER_USERNAME` and `DOCKER_PASSWORD` (use a Docker Hub access token, not the account password), push tags `${{ github.sha }}` and `latest`.
- Gate the push step with `if: github.event_name == 'push' && github.ref == 'refs/heads/main'`.

Failure points to test before opening the PR:

1. Container starts but `/ready` is 503: expected with an empty volume. Seed the volume and confirm it flips.
2. `PermissionError` on the volume: uid 10001 cannot write a root-owned volume. Chown in the image or `user:` in Compose.
3. `ModuleNotFoundError: src`: build context is not the repo root.
4. Worker boot failure on `/dev/shm`: ensure the container has it (default Docker does).
5. Postgres not ready when Airflow starts: missing healthcheck dependency.
6. CI push step runs on PRs: verify the `if` guard by opening a draft PR.
7. Image size above 200 MB: check `docker images`, confirm only numpy and Flask are installed.

Definition of done: `docker compose up`, then `curl /ready`, `/predict` return 200 with a seeded volume. CI is green on the PR. Image visible in the registry after merge.

### Part C: ETL, training, DAG, KFP (branch `part-c/airflow-kfp`)

Files: `src/etl_fetch.py`, `src/train_forecast.py`, `dags/btc_etl_pipeline.py`, `kubeflow/btc_kfp_pipeline.py`, `requirements-train.txt`, `Dockerfile.train`.

- ETL is idempotent: write `data/raw/btc_YYYY-MM-DD.parquet` keyed by the logical date, upsert by date, drop the partial current-day candle.
- Train calls `fit_ridge`, runs `evaluate_holdout`, applies the promotion rule (section 1, item 7), then `write_version` and `prune_versions(keep=5)`.
- DAG tasks: fetch, preprocess, train and evaluate, reload hint. Tasks pass file paths, never dataframes through XCom. `catchup=False`, retries with backoff, `max_active_runs=1`. The reload task tolerates failure because polling converges anyway.
- KFP: container components, `compiler.Compiler().compile(...)` to YAML. Commit the YAML. Do not deploy.

### Part D: Kubernetes (branch `part-d/k8s`)

Files: `k8s/{configmap-secret,pvc,deployment,service,hpa,ingress}.yaml`, `k8s/train-job.yaml`.

- Single-node Minikube. Enable `ingress` and `metrics-server` addons first.
- Probes: liveness `/health`, readiness `/ready`, `initialDelaySeconds: 15`, `periodSeconds: 10`.
- `securityContext`: `runAsNonRoot`, `readOnlyRootFilesystem` with an `emptyDir` for `/dev/shm`, `fsGroup: 10001` so the PVC is writable.
- Secret holds `RELOAD_TOKEN` as a placeholder. Real value via `kubectl create secret`.
- Evidence to capture: `get pods,svc,hpa,ingress`, Locust run with `kubectl get hpa --watch`.

### Part E: Verification and report

Pod delete for self-healing, manual DAG trigger, live `curl` to the ingress, 4 to 6 page PDF. Capture screenshots as each part lands, not at the end.

## 5. Branch and review workflow

One branch per part, PR to `main`, teammate reviews and merges before the next part starts. Each PR must have green CI (from Part B onward) and the part's definition of done checked.
