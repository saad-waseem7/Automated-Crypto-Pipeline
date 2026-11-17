"""Write a model version trained on SYNTHETIC prices for local smoke tests.

This exists so the API can be exercised, and the container smoke-tested, without
running the real ETL and training jobs. The metrics file is tagged
``data_source: synthetic`` so no one mistakes it for a real forecast.

Usage: python -m scripts.seed_artifacts --model-dir models
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from src.artifacts import write_version
from src.forecast_core import FloatArray, HORIZON, evaluate_holdout, fit_ridge

EVAL_ORIGINS = 60


def synthetic_closes(n_days: int, seed: int) -> FloatArray:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=0.0005, scale=0.025, size=n_days)
    prices: FloatArray = 30_000.0 * np.exp(np.cumsum(log_returns))
    return prices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--days", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    closes = synthetic_closes(args.days, args.seed)
    train = closes[: -(EVAL_ORIGINS + HORIZON)]
    model = fit_ridge(train)
    metrics = evaluate_holdout(model, closes, EVAL_ORIGINS)
    metrics["data_source"] = "synthetic"

    final_model = fit_ridge(closes)
    last_date = date.today() - timedelta(days=1)
    version = write_version(args.model_dir, final_model, closes, last_date, metrics)
    print(f"wrote version {version} to {args.model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
