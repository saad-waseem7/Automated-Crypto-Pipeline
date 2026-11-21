#!/usr/bin/env bash
# Bring the local environment up or down WITHOUT downloading or building anything.
#
#   scripts/env.sh start    Docker Compose stack, then Minikube and the cluster
#   scripts/env.sh stop     stop both, keeping every volume, image and the cluster
#   scripts/env.sh status   what is running
#
# Everything here uses images and state that already exist. If something is
# missing the script says which one-time command creates it and stops, it never
# falls back to a pull or a build behind your back.
#   K8S=0 scripts/env.sh start      Compose only
set -euo pipefail

cd "$(dirname "$0")/.."
NS=btc-forecast
K8S="${K8S:-1}"
ENV_FILE="${ENV_FILE:-.env}"
HOSTS_FILE="${HOSTS_FILE:-/etc/hosts}"
HOST_NAME=btc-forecast.local
# Images Compose needs locally. busybox only runs once per start (volume setup).
REQUIRED_IMAGES=(btc-forecast-api:local btc-forecast-airflow:local postgres:16-alpine busybox:1.37)

die() { echo "error: $*" >&2; exit 1; }
say() { printf '\n== %s\n' "$*"; }

compose() { docker compose --env-file "$ENV_FILE" --profile airflow "$@"; }

check_docker() {
  docker info >/dev/null 2>&1 \
    || die "Docker is not reachable. Start it with: sudo systemctl start docker"
}

start_compose() {
  say "Docker Compose (api, postgres, airflow)"
  [[ -f "$ENV_FILE" ]] || die "$ENV_FILE is missing. Run: cp .env.example .env and edit it"
  local missing=()
  for image in "${REQUIRED_IMAGES[@]}"; do
    docker image inspect "$image" >/dev/null 2>&1 || missing+=("$image")
  done
  if ((${#missing[@]})); then
    die "images not present locally: ${missing[*]}. One-time setup (downloads and builds): make build"
  fi
  # --pull never and --no-build: start only what is already on this machine.
  compose up -d --pull never --no-build
}

minikube_state() {
  minikube status -f '{{.Host}}' 2>/dev/null || true
}

start_kubernetes() {
  say "Minikube and the btc-forecast cluster"
  command -v minikube >/dev/null || die "minikube not found (set K8S=0 to skip Kubernetes)"
  command -v kubectl >/dev/null || die "kubectl not found (set K8S=0 to skip Kubernetes)"
  case "$(minikube_state)" in
    Running) ;;
    # A stopped cluster restarts from its saved config and cached images.
    Stopped) minikube start ;;
    *) die "no Minikube cluster yet. One-time setup (downloads images): make k8s-up" ;;
  esac
  kubectl -n "$NS" get deployment btc-forecast >/dev/null 2>&1 \
    || die "the btc-forecast deployment is missing. One-time setup: make k8s-up"
  kubectl -n ingress-nginx wait --for=condition=ready pod \
    -l app.kubernetes.io/component=controller --timeout=180s >/dev/null
  kubectl -n "$NS" wait --for=condition=ready pod \
    -l app.kubernetes.io/name=btc-forecast --timeout=180s >/dev/null
  check_hosts_entry
  kubectl -n "$NS" get pods,hpa
}

# The Minikube address can change between boots; /etc/hosts must follow it.
check_hosts_entry() {
  local ip entry
  ip="$(minikube ip)"
  entry="$(grep -E "[[:space:]]${HOST_NAME}\$" "$HOSTS_FILE" | awk '{print $1}' | head -n1 || true)"
  if [[ "$entry" != "$ip" ]]; then
    echo "note: /etc/hosts maps ${HOST_NAME} to '${entry:-nothing}', Minikube is at ${ip}. Fix with:"
    echo "  sudo sed -i '/${HOST_NAME}/d' /etc/hosts && echo \"${ip} ${HOST_NAME}\" | sudo tee -a /etc/hosts"
  fi
}

show_urls() {
  say "Ready"
  echo "Compose API  http://127.0.0.1:5000   Airflow UI  http://127.0.0.1:8080"
  [[ "$K8S" == "1" ]] && echo "Kubernetes   http://${HOST_NAME}"
  return 0
}

cmd_start() {
  check_docker
  start_compose
  [[ "$K8S" == "1" ]] && start_kubernetes
  show_urls
}

cmd_stop() {
  check_docker
  say "Stopping Compose (volumes and images are kept)"
  compose stop
  if [[ "$K8S" == "1" ]] && command -v minikube >/dev/null; then
    say "Stopping Minikube (cluster and cached images are kept)"
    minikube stop
  fi
}

cmd_status() {
  check_docker
  say "Docker Compose"
  compose ps
  if [[ "$K8S" == "1" ]] && command -v minikube >/dev/null; then
    say "Minikube: $(minikube_state)"
    kubectl -n "$NS" get pods,hpa 2>/dev/null || echo "cluster not reachable"
  fi
}

case "${1:-}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
