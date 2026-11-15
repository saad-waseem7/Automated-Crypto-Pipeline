"""Clean the raw candles and compute technical indicators and training targets.

Output (one row per UTC day, no gaps) is the single input of the trainer::

    date, close, volume, log_return, ma_7, ma_30, vol_30, y_h1 .. y_h7

``y_hN`` is log(close[t+N] / close[t]), the exact target the forecaster learns,
and is empty for the last N rows. The indicators (ma_7, ma_30, vol_30) document
and sanity-check the series. The ridge model itself uses lagged log-returns only,
because moving averages of past closes add no information beyond them.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.etl_fetch import FetchError, read_raw
from src.forecast_core import HORIZON, FloatArray

log = logging.getLogger("features")

MAX_FILL_DAYS = 3
FEATURE_FIELDS = (
    "date",
    "close",
    "volume",
    "log_return",
    "ma_7",
    "ma_30",
    "vol_30",
    *[f"y_h{h}" for h in range(1, HORIZON + 1)],
)


class FeatureError(RuntimeError):
    """Raised when the series is unusable (for example a large gap)."""


@dataclass(frozen=True)
class Series:
    days: list[date]
    close: FloatArray
    volume: FloatArray
    filled: int


def fill_gaps(
    days: list[date],
    close: FloatArray,
    volume: FloatArray,
    max_gap: int = MAX_FILL_DAYS,
) -> Series:
    """Reindex to consecutive days, forward-filling short gaps.

    A gap longer than ``max_gap`` days means missing data that forward-filling
    would invent, so it is an error rather than something to paper over.
    """
    if not days:
        raise FeatureError("no data")
    out_days: list[date] = []
    out_close: list[float] = []
    out_volume: list[float] = []
    filled = 0
    for i, day in enumerate(days):
        if i > 0:
            missing = (day - days[i - 1]).days - 1
            if missing < 0:
                raise FeatureError(f"dates not strictly increasing at {day}")
            if missing > max_gap:
                raise FeatureError(
                    f"gap of {missing} days before {day} exceeds {max_gap}"
                )
            for k in range(1, missing + 1):
                out_days.append(days[i - 1] + timedelta(days=k))
                out_close.append(out_close[-1])
                out_volume.append(0.0)
                filled += 1
        out_days.append(day)
        out_close.append(float(close[i]))
        out_volume.append(float(volume[i]))
    return Series(out_days, np.array(out_close), np.array(out_volume), filled)


def _rolling(values: FloatArray, window: int, fn: str) -> FloatArray:
    out = np.full(values.shape, np.nan)
    if values.size >= window:
        view = sliding_window_view(values, window)
        stat = view.mean(axis=1) if fn == "mean" else view.std(axis=1, ddof=1)
        out[window - 1 :] = stat
    return out


def compute_features(close: FloatArray) -> dict[str, FloatArray]:
    """Indicators and targets, aligned with ``close``. NaN marks undefined rows."""
    log_close = np.log(close)
    log_return = np.full(close.shape, np.nan)
    log_return[1:] = np.diff(log_close)
    # Rolling std over returns skips the leading NaN, so it starts at index 30.
    vol_30 = np.full(close.shape, np.nan)
    vol_30[1:] = _rolling(log_return[1:], 30, "std")
    features: dict[str, FloatArray] = {
        "log_return": log_return,
        "ma_7": _rolling(close, 7, "mean"),
        "ma_30": _rolling(close, 30, "mean"),
        "vol_30": vol_30,
    }
    for h in range(1, HORIZON + 1):
        target = np.full(close.shape, np.nan)
        if close.size > h:
            target[:-h] = log_close[h:] - log_close[:-h]
        features[f"y_h{h}"] = target
    return features


def _fmt(value: float) -> str:
    return "" if math.isnan(value) else repr(float(value))


def write_features(path: Path, series: Series, features: dict[str, FloatArray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".features-", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(FEATURE_FIELDS)
            for i, day in enumerate(series.days):
                row = [day.isoformat(), repr(float(series.close[i]))]
                row.append(repr(float(series.volume[i])))
                row.extend(_fmt(features[name][i]) for name in FEATURE_FIELDS[3:])
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_closes(path: Path) -> tuple[list[date], FloatArray]:
    """Load dates and closes from a features file, verifying there are no gaps."""
    days: list[date] = []
    closes: list[float] = []
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                days.append(date.fromisoformat(record["date"]))
                closes.append(float(record["close"]))
    except (KeyError, ValueError, OSError) as exc:
        raise FeatureError(f"cannot read features file {path}: {exc}") from exc
    if not days:
        raise FeatureError(f"{path} is empty")
    for earlier, later in zip(days, days[1:]):
        if (later - earlier).days != 1:
            raise FeatureError(f"features file has a gap between {earlier} and {later}")
    return days, np.array(closes)


def run(raw_path: Path, out_path: Path) -> Series:
    candles = read_raw(raw_path)
    if not candles:
        raise FeatureError(f"no raw data in {raw_path}")
    days = sorted(candles)
    close = np.array([candles[d].close for d in days])
    volume = np.array([candles[d].volume for d in days])
    series = fill_gaps(days, close, volume)
    write_features(out_path, series, compute_features(series.close))
    log.info(
        "wrote %d rows (%d gap days forward-filled), last day %s",
        len(series.days),
        series.filled,
        series.days[-1],
    )
    return series


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=Path(os.environ.get("BTC_RAW_PATH", "data/raw/btc_usdt_1d.csv")),
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path(
            os.environ.get("BTC_FEATURES_PATH", "data/processed/features.csv")
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        run(args.raw_path, args.out_path)
    except (FeatureError, FetchError) as exc:
        log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
