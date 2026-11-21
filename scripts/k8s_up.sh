#!/usr/bin/env bash
# Deploy the stack to the current kubectl context (Minikube by default).
# Idempotent: safe to re-run. Generates the reload token once and never commits it.
set -euo pipefail

cd "$(dirname "$0")/.."
NS=btc-forecast

command -v kubectl >/dev/null || { echo "kubectl not found" >&2; exit 1; }

if [[ "${USE_MINIKUBE:-1}" == "1" ]]; then
  command -v minikube >/dev/null || { echo "minikube not found" >&2; exit 1; }
  # Start only when needed. A stopped cluster restarts from its saved config and
  # cached images, so this downloads nothing after the first run.
  if [[ "$(minikube status -f '{{.Host}}' 2>/dev/null || true)" != "Running" ]]; then
    minikube start --cpus=4 --memory=4096
  fi
  # Enable each addon once. Re-enabling an installed addon re-applies its
  # manifests for no benefit, and its readiness check times out on slow pulls
  # even though the pods come up later, so never abort on it.
  if ! kubectl -n ingress-nginx get deployment ingress-nginx-controller >/dev/null 2>&1; then
    minikube addons enable ingress || echo "ingress addon check timed out, waiting for the controller" >&2
  fi
  if ! kubectl -n kube-system get deployment metrics-server >/dev/null 2>&1; then
    minikube addons enable metrics-server || echo "metrics-server addon check timed out" >&2
  fi
  kubectl -n ingress-nginx wait --for=condition=ready pod \
    -l app.kubernetes.io/component=controller --timeout=600s \
    || { kubectl -n ingress-nginx get pods >&2; echo "ingress controller not ready" >&2; exit 1; }
fi

kubectl apply -f k8s/namespace.yaml

# Keep an existing reload token across re-runs: applying k8s/configmap-secret.yaml
# resets it to the committed empty value, so read it first and restore it after.
token="$(kubectl -n "$NS" get secret btc-forecast-secret \
  -o jsonpath='{.data.RELOAD_TOKEN}' 2>/dev/null | base64 -d 2>/dev/null || true)"
fresh=0
if [[ -z "$token" ]]; then
  token="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  fresh=1
fi

kubectl apply -k k8s
# The patch goes through stdin so the token never appears in the process list.
kubectl -n "$NS" patch secret btc-forecast-secret --type merge --patch-file /dev/stdin \
  <<<"{\"stringData\":{\"RELOAD_TOKEN\":\"${token}\"}}"
# Pods read the token at start, so only a newly generated one needs a restart.
if [[ "$fresh" == "1" ]]; then
  kubectl -n "$NS" rollout restart deployment/btc-forecast
fi

# A fresh volume has no model, and pods stay unready until one exists. Train once
# in that case only: on every later run the model is already on the volume, and
# a needless job would re-download candles and retrain.
kubectl -n "$NS" wait --for=jsonpath='{.status.phase}'=Running pod \
  -l app.kubernetes.io/name=btc-forecast --timeout=300s
if kubectl -n "$NS" exec deployment/btc-forecast -- test -e /app/models/CURRENT; then
  echo "model already on the volume, skipping the bootstrap training job"
else
  job="btc-train-bootstrap-$(date +%s)"
  kubectl -n "$NS" create job "$job" --from=cronjob/btc-train
  kubectl -n "$NS" wait --for=condition=complete "job/$job" --timeout=600s \
    || { kubectl -n "$NS" logs "job/$job" --tail=50 >&2; exit 1; }
fi
kubectl -n "$NS" rollout status deployment/btc-forecast --timeout=300s

kubectl -n "$NS" get pods,svc,hpa,ingress
if [[ "${USE_MINIKUBE:-1}" == "1" ]]; then
  ip="$(minikube ip)"
  echo
  echo "NodePort: curl http://${ip}:30080/ready"
  echo "Ingress:  curl -H 'Host: btc-forecast.local' http://${ip}/ready"
fi
