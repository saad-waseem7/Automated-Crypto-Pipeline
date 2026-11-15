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
| 7 | Promotion on "threshold criteria" with a 7-day test window (C1) | n=7 of a near random walk is noise. An absolute MAPE threshold either always passes or never does. | Backtest over 60 origins, scored against a naive baseline on the same origins. Promote only if the candidate is not worse than the baseline by more than 5 percent (relative MAPE). No incumbent comparison: overlapping windows would favor the incumbent, which saw part of the data. |
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

### Part B: Docker, Compose, CI/CD (branch `part-b/containers-ci`, implemented)

Files: `app/Dockerfile`, `.dockerignore`, `docker-compose.yml`, `.env.example`, `.github/workflows/ci_cd.yml`, `.github/dependabot.yml`, `scripts/check_api.sh`, `scripts/smoke_test.sh`, `dags/.gitkeep`.

Decisions:

- Registry is GHCR, authenticated with the built-in `GITHUB_TOKEN`. No long-lived password in repo secrets. Switching to Docker Hub means changing the login step to `DOCKER_USERNAME` and a Docker Hub access token (never the account password) and the `IMAGE` env. After the first push, set the package visibility to public in GitHub (Packages, Package settings) or the "public link" deliverable fails.
- Dockerfile stages: `builder`, `base`, `seed` (synthetic model generator, for CI and local only), `runtime` (last, so it is the default target). Non-root uid 10001, exec-form CMD, HEALTHCHECK without curl, build context is the repo root.
- Compose: `api` mounts the model volume read-only, with a read-only root filesystem, all capabilities dropped and `no-new-privileges`. `seed` and the Airflow stack sit behind profiles. Airflow runs as a single `standalone` container to fit a 16 GB laptop. Training dependencies are not installed in it.
- CI: `quality` job (black, flake8, mypy, pytest with coverage, shellcheck, compose config) then `image` job (build, container smoke test, push only on `main`). PRs never push. Tags: full commit SHA and `latest`.
- Smoke test checks uid 10001, cold start (`/health` 200, `/ready` 503 with an empty volume), warm start after seeding, and a limit of 250 MB on the filesystem size inside the image (measured with du, because `docker image inspect` varies by image store).

What was verified before opening the PR (in a sandbox that cannot pull base images):

| Check | Result |
|-------|--------|
| hadolint on the Dockerfile, actionlint on the workflow, shellcheck on both scripts | clean |
| `docker compose config` with `.env.example`, and a missing secret fails loudly | pass |
| Real build context via a `FROM scratch` build | only `app/`, `src/`, `scripts/*.py` |
| Container layout run from a clean venv with runtime-only dependencies, exact CMD, no `PYTHONPATH` | pass |
| `check_api.sh` against a live gunicorn, and against a broken API | passes, fails as it should |
| Part A fix found while designing the volume flow: artifacts were mode 0700 and unreadable by other users | fixed, with a test |

Not verified (needs your machine or the CI run): the actual `docker build` of both images, `scripts/smoke_test.sh`, and the `apache/airflow:3.1.8` image with `standalone` and the Postgres connection. The first real run of these is the PR's CI job. If it fails, paste the log.

Failure points to check on the PR:

1. `/ready` is 503 right after `docker compose up`: expected until the volume is seeded. Run `docker compose run --rm seed` first.
2. `PermissionError` on the volume: a fresh named volume copies the image's ownership only on its first writable mount. Seed before the first `up`.
3. `ModuleNotFoundError: src`: the build context is not the repo root.
4. Compose errors about missing variables: copy `.env.example` to `.env` first. Interpolation is file-wide, so Airflow secrets are required even for API-only use.
5. Airflow container restarts: check `docker compose logs airflow`. First suspects are the Fernet key format and the Postgres password.
6. CI push step running on a PR: it must not. Confirm the `image` job on the PR shows the push steps as skipped.
7. Image over the size limit: the smoke test prints the filesystem size and the layer breakdown.

Definition of done: CI green on the PR, `docker compose up` gives a working API with a seeded volume, and after merge the image is visible in GHCR.

### Part C: ETL, training, DAG, KFP (branch `part-c/etl-train-dags`, implemented)

Files: `src/etl_fetch.py`, `src/features.py`, `src/train_forecast.py`, `dags/btc_etl_pipeline.py`, `kubeflow/btc_kfp_pipeline.py` and its compiled `.yaml`, `airflow/Dockerfile`, `scripts/fake_binance.py`, `requirements-kfp.txt`, compose and CI changes.

Data flow: Binance klines, `raw/btc_usdt_1d.csv`, `processed/features.csv`, `models/versions/<v>` plus `CURRENT`, then the API (by polling, plus a best-effort reload call).

Decisions and deviations:

- Data source is Binance public klines only, with `data-api.binance.vision` as the fallback host. yfinance was dropped from the plan: it adds a heavy dependency tree and is an unofficial scraper. The assignment accepts any one source.
- Only closed candles are stored (the open one is dropped), the raw CSV is upserted by UTC date, and writes are atomic. Re-running is safe.
- Features (MA7, MA30, volatility, targets) are computed as the assignment requires and written to `features.csv`. The ridge model itself uses lagged log-returns only, so the indicators document and sanity-check the series but are not model inputs.
- Training fits a candidate on data before the last 60 forecast origins, scores it against the naive baseline on those origins, and promotes only if it is not worse than the baseline by more than 5 percent. It then refits on all data for serving. A rejected model publishes nothing and the old one keeps serving. Both outcomes exit 0 and write a JSON report; only unusable data is a failure.
- Airflow tasks run our scripts with a separate venv (numpy only) baked into `airflow/Dockerfile`, not Airflow's interpreter. Airflow is pinned to 3.1.8 on Python 3.11, the same as the API image.
- The reload task is best effort. If it fails, the API still switches within its poll interval.
- Kubeflow: three container components using the same image CI publishes. The YAML is compiled and checked in CI for drift but not deployed.
- The same runtime image runs the API and the training scripts, so Part D can reuse it for a Kubernetes Job.

Verified:

| Check | Result |
|-------|--------|
| 180 pytest tests at 98 percent coverage, including fetch, features, train and serve end to end | pass |
| Real Airflow 3.1.8 `dags test` with fake Binance and a live API: all 4 tasks succeed, reload reaches the API | pass |
| Kubeflow spec compiled with kfp 2.18.0, two compiles byte-identical | pass |
| actionlint, hadolint, shellcheck, `docker compose config` | clean |

Not verified until the PR's CI runs: the Airflow image build (tag `apache/airflow:3.1.8-python3.11`), the DAG inside that image, volume permissions between uid 50000 and 10001, and the compose `airflow` profile.

Not verifiable in the sandbox: a run against the real Binance API. Run `python -m src.etl_fetch` on your machine once to confirm.

Failure points:

1. The gate rejects the model on real data: the report says `rejected` and the old model keeps serving. Check `data/reports/last_train.json`. Raising `--tolerance` is a policy choice, not a fix.
2. HTTP 451 from Binance: the fallback host is tried automatically. If both fail, the task fails and Airflow retries with backoff.
3. A gap longer than 3 days in the series stops preprocessing instead of inventing data.
4. Airflow cannot write the volumes: `init-volumes` must complete first (it runs as root and fixes ownership).
5. DAG not visible in the UI: Airflow 3 shows DAGs only after the dag-processor parses them. Wait about 30 seconds or check `docker compose logs airflow`.

### Part D: Kubernetes (branch `part-d/kubernetes`, implemented)

Files: `k8s/` (namespace, configmap, pvc, deployment, service, hpa, ingress, train-job, kustomization), `scripts/k8s_up.sh`, `loadtest/locustfile.py`, `tests/test_k8s_manifests.py`, CI job `k8s-e2e`.

- Deployment: 3 replicas, `maxUnavailable: 0`, requests 250m/256Mi, limits 500m/512Mi, non-root, read-only root, all caps dropped, seccomp RuntimeDefault, namespace enforces Pod Security `restricted`. `/dev/shm` is a memory emptyDir for gunicorn heartbeats.
- Probes: liveness `/health`, readiness `/ready` (a justified deviation from probing one endpoint: a pod without a model must not take traffic, but must not be killed either), plus a startup probe.
- HPA v2: 2 to 8 replicas, CPU 60% of request, fast scale-up, 120 s scale-down stabilisation.
- Services: ClusterIP for the Ingress (`btc-forecast.local`, NGINX) and NodePort 30080 for direct access.
- Training: CronJob `btc-train` (00:15 UTC, `Forbid` overlap) runs fetch, features, train. It writes to the shared PVC and the API pods pick the version up by polling `CURRENT`. No restart, no API call.
- Image pinned to a commit SHA in `kustomization.yaml`. The reload token is generated by `scripts/k8s_up.sh` and never committed (the Secret is deliberately not in the kustomization).
- Verified in CI on kind: pods stay unready with no model, in-cluster training succeeds, `/predict` works through the Ingress, deleting a pod causes at most 2 failed requests over 20 s, HPA reports CPU metrics. Not verified: real Minikube (no cluster in the authoring sandbox).

Known limits, stated plainly:
- RWO volume: correct on one node, breaks on multi-node. Needs RWX or object storage there.
- No cluster-side Airflow or Kubeflow deployment. Airflow runs under Compose, Kubeflow is a compiled, drift-checked spec.
- A fresh cluster serves nothing until the first training job finishes (the runtime image has no synthetic seeder, by design).
- HPA scale-up is capped by the Minikube node (4 CPUs): 8 replicas x 250m requests fit, but Locust and the node share CPU, so expect noisy results.

Runbook and evidence checklist:
1. `make k8s-up` (needs `minikube` and `kubectl`).
2. `echo "$(minikube ip) btc-forecast.local" | sudo tee -a /etc/hosts`, then `curl -X POST http://btc-forecast.local/predict`.
3. Screenshot: `kubectl -n btc-forecast get pods,svc,hpa,ingress,pvc`.
4. `pip install -r requirements-loadtest.txt && make loadtest`, with `kubectl -n btc-forecast get hpa -w` in another terminal. Screenshot replicas rising.
5. Self-healing for Part E: `kubectl -n btc-forecast delete pod <one>` while curling in a loop.

### Part E: Verification and report

Pod delete for self-healing, manual DAG trigger, live `curl` to the ingress, 4 to 6 page PDF. Capture screenshots as each part lands, not at the end.

## 5. Branch and review workflow

One branch per part, PR to `main`, teammate reviews and merges before the next part starts. Each PR must have green CI (from Part B onward) and the part's definition of done checked.
