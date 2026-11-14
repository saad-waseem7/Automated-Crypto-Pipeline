#!/usr/bin/env bash
# End-to-end container smoke test.
# Usage: scripts/smoke_test.sh <runtime_image> <seed_image>
#
# Verifies, against the real images:
#   1. the container runs as uid 10001 with a read-only root filesystem
#   2. cold start: no model on the volume -> /health 200 but /ready 503
#   3. after seeding the volume the API becomes ready and answers correctly
#   4. the image stays under MAX_IMAGE_MB (default 300)
set -euo pipefail

RUNTIME_IMAGE="${1:?usage: smoke_test.sh <runtime_image> <seed_image>}"
SEED_IMAGE="${2:?usage: smoke_test.sh <runtime_image> <seed_image>}"
PORT="${SMOKE_PORT:-5099}"
MAX_IMAGE_MB="${MAX_IMAGE_MB:-300}"
SUFFIX="$$"
VOLUME="btc-smoke-models-${SUFFIX}"
COLD="btc-smoke-cold-${SUFFIX}"
WARM="btc-smoke-warm-${SUFFIX}"
HERE="$(cd "$(dirname "$0")" && pwd)"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

cleanup() {
    docker rm -f "$COLD" "$WARM" >/dev/null 2>&1 || true
    docker volume rm "$VOLUME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

run_api() { # name
    docker run -d --name "$1" -p "127.0.0.1:${PORT}:5000" \
        -v "${VOLUME}:/app/models:ro" \
        --read-only --cap-drop ALL --security-opt no-new-privileges:true \
        -e RELOAD_TOKEN=smoke-token-0123456789 -e MODEL_POLL_SECONDS=1 \
        "$RUNTIME_IMAGE" >/dev/null
}

wait_for_http() { # url
    for _ in $(seq 1 30); do
        curl -s -o /dev/null "$1" && return 0
        sleep 1
    done
    docker logs "$2" >&2 || true
    fail "$1 never answered"
}

size_mb=$(($(docker image inspect -f '{{.Size}}' "$RUNTIME_IMAGE") / 1000000))
echo "image size: ${size_mb} MB (limit ${MAX_IMAGE_MB})"
[ "$size_mb" -le "$MAX_IMAGE_MB" ] || fail "image is ${size_mb} MB, over ${MAX_IMAGE_MB}"

docker volume create "$VOLUME" >/dev/null
# A fresh named volume copies the image's /app/models ownership only on its first
# writable mount. Do that first, otherwise a read-only first mount can leave the
# volume root-owned and the seed job (uid 10001) could not write to it.
docker run --rm -v "${VOLUME}:/app/models" "$SEED_IMAGE" true

echo "== cold start (empty volume)"
run_api "$COLD"
wait_for_http "http://127.0.0.1:${PORT}/health" "$COLD"
uid=$(docker exec "$COLD" id -u)
[ "$uid" = "10001" ] || fail "container runs as uid $uid, expected 10001"
health=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health")
ready=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/ready")
[ "$health" = "200" ] || fail "/health returned $health on cold start, expected 200"
[ "$ready" = "503" ] || fail "/ready returned $ready on cold start, expected 503"
docker rm -f "$COLD" >/dev/null

echo "== seeding the volume"
docker run --rm -v "${VOLUME}:/app/models" "$SEED_IMAGE"

echo "== warm start"
run_api "$WARM"
wait_for_http "http://127.0.0.1:${PORT}/health" "$WARM"
"$HERE/check_api.sh" "http://127.0.0.1:${PORT}" || {
    docker logs "$WARM" >&2
    exit 1
}

echo "OK: container smoke test passed"
