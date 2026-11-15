"""Train the 7-day forecaster, evaluate it honestly, and promote it only if it passes.

Procedure:
1. Hold out the last ``eval_origins`` forecast origins. Fit a candidate on data
   strictly before that window and score it there next to a naive "tomorrow
   equals today" baseline.
2. Gate: promote only if the candidate is not worse than the baseline by more
   than ``tolerance`` (relative MAPE). On a near random walk like BTC, a fixed
   absolute MAPE threshold would either always pass or never pass.
3. If it passes, refit on all data and publish a new artifact version, then
   prune old versions. If not, publish nothing and keep serving the old model.

The outcome is written to a JSON report so an orchestrator can branch on it.
The process exits 0 for both outcomes, and non-zero only when data is unusable.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from src.artifacts import ArtifactError, prune_versions, write_version
from src.features import FeatureError, read_closes
from src.forecast_core import (
    HORIZON,
    FloatArray,
    ForecastError,
    evaluate_holdout,
    fit_ridge,
)

log = logging.getLogger("train_forecast")

DEFAULT_EVAL_ORIGINS = 60
DEFAULT_TOLERANCE = 0.05
DEFAULT_KEEP = 5


@dataclass(frozen=True)
class TrainResult:
    promoted: bool
    reason: str
    version: str | None
    metrics: dict[str, Any]
    data_last_date: date
    n_closes: int


def passes_gate(metrics: dict[str, Any], tolerance: float) -> tuple[bool, str]:
    model_mape = metrics["mape"]
    naive_mape = metrics["baseline_naive"]["mape"]
    if not (math.isfinite(model_mape) and math.isfinite(naive_mape)):
        return False, "non-finite MAPE"
    limit = naive_mape * (1.0 + tolerance)
    if model_mape <= limit:
        return True, f"MAPE {model_mape:.3f} <= {limit:.3f} (naive {naive_mape:.3f})"
    return False, f"MAPE {model_mape:.3f} > {limit:.3f} (naive {naive_mape:.3f})"


def train(
    days: list[date],
    closes: FloatArray,
    model_dir: Path,
    *,
    eval_origins: int = DEFAULT_EVAL_ORIGINS,
    tolerance: float = DEFAULT_TOLERANCE,
    keep: int = DEFAULT_KEEP,
    data_source: str = "binance:BTCUSDT",
) -> TrainResult:
    first_origin = closes.size - HORIZON - eval_origins
    if first_origin < 1:
        raise ForecastError("not enough data for the requested evaluation window")
    candidate = fit_ridge(closes[: first_origin + 1])
    metrics = evaluate_holdout(candidate, closes, eval_origins)
    ok, reason = passes_gate(metrics, tolerance)
    log.info("gate %s: %s", "PASSED" if ok else "FAILED", reason)
    if not ok:
        return TrainResult(False, reason, None, metrics, days[-1], closes.size)

    metrics.update(
        {
            "data_source": data_source,
            "evaluated_on": f"last {eval_origins} origins, model fit before window",
            "gate": reason,
            "n_train_closes": int(closes.size),
        }
    )
    final_model = fit_ridge(closes)
    version = write_version(model_dir, final_model, closes, days[-1], metrics)
    removed = prune_versions(model_dir, keep)
    log.info("promoted version %s, pruned %d old versions", version, len(removed))
    return TrainResult(True, reason, version, metrics, days[-1], closes.size)


def write_report(path: Path, result: TrainResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "promoted" if result.promoted else "rejected",
        "reason": result.reason,
        "version": result.version,
        "data_last_date": result.data_last_date.isoformat(),
        "n_closes": result.n_closes,
        "metrics": {
            k: result.metrics[k]
            for k in ("mae", "rmse", "mape", "baseline_naive", "n_origins")
        },
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    fd, tmp = tempfile.mkstemp(prefix=".report-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    env = os.environ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path(env.get("BTC_FEATURES_PATH", "data/processed/features.csv")),
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path(env.get("MODEL_DIR", "models"))
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path(env.get("BTC_REPORT_PATH", "data/reports/last_train.json")),
    )
    parser.add_argument("--eval-origins", type=int, default=DEFAULT_EVAL_ORIGINS)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=env.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        days, closes = read_closes(args.features_path)
        result = train(
            days,
            closes,
            args.model_dir,
            eval_origins=args.eval_origins,
            tolerance=args.tolerance,
            keep=args.keep,
        )
        write_report(args.report_path, result)
    except (FeatureError, ForecastError, ArtifactError, OSError) as exc:
        log.error("%s", exc)
        return 2
    print(f"{'promoted' if result.promoted else 'rejected'}: {result.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
