#!/usr/bin/env bash
# Black-box assertions against a running API that has a model loaded.
# Usage: scripts/check_api.sh [base_url]     (default http://127.0.0.1:5000)
set -euo pipefail

BASE="${1:-http://127.0.0.1:5000}"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

# JSON assertion helper: reads stdin as JSON, evaluates $1 with it bound to `d`.
expect() {
    python3 -c 'import json, sys; d = json.load(sys.stdin); assert eval(sys.argv[1]), d' "$1" ||
        fail "assertion failed: $1"
}

code=000
for _ in $(seq 1 30); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/ready" || true)
    [ "$code" = "200" ] && break
    sleep 1
done
[ "$code" = "200" ] || fail "/ready did not return 200 within 30s (last: $code)"

curl -fsS "$BASE/health" |
    expect 'd["status"] == "ok" and d["model_loaded"] is True'

curl -fsS -X POST "$BASE/predict" |
    expect 'len(d["predictions"]) == 7 and all(p["predicted_close"] > 0 for p in d["predictions"])'

curl -fsS -X POST "$BASE/predict" -H 'Content-Type: application/json' \
    -d '{"forecast_days": 3}' |
    expect 'd["forecast_days"] == 3 and len(d["predictions"]) == 3'

curl -fsS "$BASE/metrics" |
    expect 'all(k in d["metrics"] for k in ("mae", "rmse", "mape"))'

bad=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/predict" \
    -H 'Content-Type: application/json' -d '{"forecast_days": 99}')
[ "$bad" = "400" ] || fail "invalid forecast_days returned $bad, expected 400"

echo "OK: all API checks passed against $BASE"
