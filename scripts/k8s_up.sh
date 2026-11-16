#!/usr/bin/env bash
# Deploy the stack to the current kubectl context (Minikube by default).
# Idempotent: safe to re-run. Generates the reload token once and never commits it.
set -euo pipefail

cd "$(dirname "$0")/.."
NS=btc-forecast

command -v kubectl >/dev/null || { echo "kubectl not found" >&2; exit 1; }

if [[ "${USE_MINIKUBE:-1}" == "1" ]]; then
  command -v minikube >/dev/null || { echo "minikube not found" >&2; exit 1; }
  minikube status >/dev/null 2>&1 || minikube start --cpus=4 --memory=4096
  # The addon's own readiness check times out on slow image pulls even though
  # the pods come up later, so do not abort on it. Wait explicitly below.
  minikube addons enable ingress || echo "ingress addon check timed out, waiting for the controller" >&2
  minikube addons enable metrics-server || echo "metrics-server addon check timed out" >&2
  kubectl -n ingress-nginx wait --for=condition=ready pod \
    -l app.kubernetes.io/component=controller --timeout=600s \
    || { kubectl -n ingress-nginx get pods >&2; echo "ingress controller not ready" >&2; exit 1; }
fi

kubectl apply -f k8s/namespace.yaml
if ! kubectl -n "$NS" get secret btc-forecast-secret >/dev/null 2>&1; then
  token="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  kubectl -n "$NS" create secret generic btc-forecast-secret \
    --from-literal=RELOAD_TOKEN="$token"
fi

kubectl apply -k k8s

# No model exists on a fresh volume: run one training job now instead of waiting
# for the nightly schedule. Pods stay Ready=false until it finishes.
job="btc-train-bootstrap-$(date +%s)"
kubectl -n "$NS" create job "$job" --from=cronjob/btc-train
kubectl -n "$NS" wait --for=condition=complete "job/$job" --timeout=600s \
  || { kubectl -n "$NS" logs "job/$job" --tail=50 >&2; exit 1; }
kubectl -n "$NS" rollout status deployment/btc-forecast --timeout=300s

kubectl -n "$NS" get pods,svc,hpa,ingress
if [[ "${USE_MINIKUBE:-1}" == "1" ]]; then
  ip="$(minikube ip)"
  echo
  echo "NodePort: curl http://${ip}:30080/ready"
  echo "Ingress:  curl -H 'Host: btc-forecast.local' http://${ip}/ready"
fi
