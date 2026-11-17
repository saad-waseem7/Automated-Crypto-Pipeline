# Operations

Runbooks for running, testing and troubleshooting. For how the system works, see
[ARCHITECTURE.md](ARCHITECTURE.md).

## After a reboot

Docker starts by itself. Compose services marked `restart: unless-stopped` normally come back.
Minikube does not.

```bash
cd Automated-Crypto-Pipeline && source .venv/bin/activate
docker compose --profile airflow up -d      # API, Postgres, Airflow
minikube start                              # reuses the existing cluster
kubectl -n btc-forecast get pods            # wait for Running 1/1
grep btc-forecast /etc/hosts; minikube ip   # the two addresses must match
```

If the addresses differ: `sudo sed -i '/btc-forecast.local/d' /etc/hosts`, then
`echo "$(minikube ip) btc-forecast.local" | sudo tee -a /etc/hosts`.
If the cluster or deployment is gone, `make k8s-up` rebuilds everything and is safe to re-run.

Stop cleanly before shutdown: `minikube stop` and `docker compose --profile airflow stop`.
Never run `docker compose down -v` unless you want to delete the models and Airflow data.

## Local development

```bash
make install     # .venv with Python 3.11 (override: make install PYTHON=/path/to/python3.11)
make check       # black, flake8, mypy strict, pytest with coverage
make seed        # SYNTHETIC model into ./models, for smoke tests only
make run         # gunicorn on :5000
make help        # all targets
```

## Docker Compose

```bash
cp .env.example .env                        # then change every value
docker compose run --rm seed                # first run only: SYNTHETIC model so /ready is 200
docker compose up -d --build                # API on http://127.0.0.1:5000
scripts/check_api.sh                        # black-box checks against the running API
docker compose --profile airflow up -d      # adds Postgres and Airflow on http://127.0.0.1:8080
docker compose exec airflow airflow dags trigger btc_etl_pipeline
make smoke                                  # container smoke test (uid, cold start, size)
```

Ports are bound to 127.0.0.1 only. Airflow replaces the synthetic model with a real one after its
first successful run.

## Training pipeline without Airflow

```bash
# offline, synthetic prices (what CI does)
python -m scripts.fake_binance --port 8765 &
export BINANCE_BASE_URLS=http://127.0.0.1:8765
python -m src.etl_fetch && python -m src.features && python -m src.train_forecast

# real data, no API key
unset BINANCE_BASE_URLS
python -m src.etl_fetch && python -m src.features && python -m src.train_forecast
cat data/reports/last_train.json            # promoted or rejected, with the metrics
```

Kubeflow spec: `pip install -r requirements/kfp.txt && python kubeflow/btc_kfp_pipeline.py`
recompiles `kubeflow/btc_kfp_pipeline.yaml`. CI fails if the committed file is stale.

## Kubernetes

Prerequisites: Docker, `kubectl`, `minikube`. The image on GHCR is public, so no pull secret is needed.

```bash
make k8s-up          # cluster, ingress and metrics-server addons, manifests, first training job
echo "$(minikube ip) btc-forecast.local" | sudo tee -a /etc/hosts
curl -X POST http://btc-forecast.local/predict
kubectl -n btc-forecast get pods,svc,hpa,ingress,pvc
make k8s-down        # delete the namespace
```

| Object | Behavior |
|--------|----------|
| Deployment `btc-forecast` | 3 replicas, rolling update with zero unavailable, liveness `/health`, readiness `/ready`, requests 100m / 128Mi, limits 500m / 512Mi |
| HPA | 2 to 8 replicas at 60% CPU of the request, fast scale-up, 120 s scale-down stabilization |
| PodDisruptionBudget | at most 1 pod unavailable during node drains |
| Services | ClusterIP (used by the ingress) and NodePort 30080 |
| Ingress | NGINX, host `btc-forecast.local` |
| CronJob `btc-train` | daily 00:15 UTC, no overlap, writes models and raw data to the PVC |
| Secret | `RELOAD_TOKEN` is empty in git. `scripts/k8s_up.sh` generates a random value and keeps it across runs. |

A fresh cluster serves nothing until the first training job finishes. Pods stay unready and are
never restarted for it. To retrain now: `kubectl -n btc-forecast create job --from=cronjob/btc-train retrain-1`.
Pods pick up the new version within `MODEL_POLL_SECONDS`, with no restart.

### Load test and autoscaling

```bash
make install-loadtest
kubectl -n btc-forecast get hpa -w          # terminal 1
make loadtest                               # terminal 2: 200 users, 3 minutes
```

Locust and the cluster share one machine, so treat latency and throughput as local numbers. The HPA
replica count is the meaningful result. Reference run on a laptop: 2 to 8 replicas in about two
minutes, 213,752 requests, 0 failures, p95 100 ms.

### Self-healing check

```bash
for i in $(seq 60); do printf '%s %s\n' "$(date +%T)" "$(curl -s -o /dev/null -w '%{http_code}' -X POST http://btc-forecast.local/predict)"; sleep 0.5; done > /tmp/heal.txt &
sleep 5
kubectl -n btc-forecast delete $(kubectl -n btc-forecast get pods -l app.kubernetes.io/name=btc-forecast -o name | head -1)
wait; awk '{print $2}' /tmp/heal.txt | sort | uniq -c
```

The Deployment recreates the pod immediately. A stray 502 or 503 is possible while the ingress
removes the terminating pod from its endpoints.

## CI

`.github/workflows/ci_cd.yml` runs on every pull request and push to `main`:

| Job | Checks |
|-----|--------|
| Lint, types, tests | black, flake8, mypy strict, pytest with coverage, shellcheck, compose config, dependency audit |
| Kubeflow spec | recompiles the pipeline and fails on drift |
| Airflow DAG end to end | real `airflow dags test` with a fake Binance server and a live API |
| Kubernetes end to end | kind cluster: unready without a model, in-cluster training, predict through the ingress, pod deletion, HPA metrics |
| Build, smoke test, publish | image build, container smoke test, push to GHCR on `main` only |

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `/ready` is 503 after `docker compose up` | No model yet. Run `docker compose run --rm seed`, or trigger the DAG. |
| `PermissionError` on the model volume | Seed before the first `up`; the `init-volumes` service must complete before Airflow starts. |
| `ModuleNotFoundError: src` in Docker | The build context is not the repository root. |
| Compose says a variable is missing | Copy `.env.example` to `.env`. Interpolation is file-wide. |
| `make k8s-up` stops at the ingress addon | Slow image pull. Check `kubectl -n ingress-nginx get pods` and re-run; it is idempotent. |
| Pods stay `0/1` | Expected until the first training job completes. Check `kubectl -n btc-forecast logs job/<name>`. |
| HPA shows `<unknown>` | metrics-server still warming up (1 to 2 minutes). Check `kubectl top pods -n btc-forecast`. |
| `ImagePullBackOff` | The GHCR package must be public. |
| Training reports `rejected` | The candidate was worse than the naive baseline by more than 5%. The old model keeps serving. See `last_train.json`. |
| Binance returns HTTP 451 | Region block. The fallback host is tried automatically; if both fail the task fails and Airflow retries. |
| A gap over 3 days stops preprocessing | By design. The pipeline refuses to invent more than 3 days of prices. |
| DAG missing in the Airflow UI | The dag-processor needs about 30 seconds. See `docker compose logs airflow`. |
