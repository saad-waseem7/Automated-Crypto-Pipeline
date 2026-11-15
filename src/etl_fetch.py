"""Fetch daily BTC/USDT candles from Binance public market data into a raw CSV.

Design points:
- Idempotent: the raw file is keyed by UTC date and upserted, so re-running a
  day, or a whole backfill, produces the same file.
- Only closed candles are stored. Binance returns the current, still open
  candle as the last row; training on its partial close would leak a wrong value.
- Standard library only, so it runs in any environment (Airflow venv, K8s Job).
- Hosts are tried in order. api.binance.com is blocked in some regions (HTTP
  451); data-api.binance.vision serves the same public market data.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

log = logging.getLogger("etl_fetch")

DEFAULT_BASES = ("https://api.binance.com", "https://data-api.binance.vision")
SYMBOL = "BTCUSDT"
INTERVAL = "1d"
PAGE_LIMIT = 1000
MAX_PAGES = 100
FIRST_DAY = date(2017, 8, 17)  # first BTCUSDT daily candle on Binance
DAY_MS = 86_400_000
CSV_FIELDS = ("date", "open", "high", "low", "close", "volume")
RETRYABLE_STATUS = {418, 429, 500, 502, 503, 504}

JsonGetter = Callable[[str], Any]


class FetchError(RuntimeError):
    """Raised when data cannot be fetched or fails validation."""


@dataclass(frozen=True)
class Candle:
    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float


def _utc_ms_now() -> int:
    return int(time.time() * 1000)


def _day_to_ms(day: date) -> int:
    return int(
        datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000
    )


def _validate(candle: Candle) -> Candle:
    values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
    if not all(math.isfinite(v) for v in values):
        raise FetchError(f"non-finite value in candle {candle.day}")
    if min(candle.open, candle.high, candle.low, candle.close) <= 0:
        raise FetchError(f"non-positive price in candle {candle.day}")
    if candle.volume < 0:
        raise FetchError(f"negative volume in candle {candle.day}")
    if candle.high < max(candle.open, candle.close) or candle.low > min(
        candle.open, candle.close
    ):
        raise FetchError(f"inconsistent high/low in candle {candle.day}")
    return candle


def parse_kline(row: Sequence[Any], now_ms: int) -> Candle | None:
    """Convert one Binance kline row. Returns None for a candle still open.

    Row layout: [openTime, open, high, low, close, volume, closeTime, ...],
    with prices and volume encoded as strings.
    """
    try:
        open_ms = int(row[0])
        close_ms = int(row[6])
        candle = Candle(
            day=datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc).date(),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )
    except (IndexError, TypeError, ValueError, OverflowError, OSError) as exc:
        raise FetchError(f"malformed kline row: {exc}") from exc
    if open_ms % DAY_MS != 0:
        raise FetchError(f"candle {open_ms} is not aligned to a UTC day")
    if close_ms >= now_ms:
        return None
    return _validate(candle)


def http_get_json(
    url: str,
    *,
    timeout: float = 15.0,
    retries: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """GET and decode JSON with exponential backoff on transient failures."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "btc-pipeline/1.0"})
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            if exc.code not in RETRYABLE_STATUS:
                raise FetchError(f"HTTP {exc.code} from {url}") from exc
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 0
            last = exc
            sleep(max(delay, 2.0**attempt))
        except (URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            sleep(2.0**attempt)
        except ValueError as exc:
            raise FetchError(f"invalid JSON from {url}") from exc
    raise FetchError(f"gave up after {retries} attempts: {last}")


def _fetch_from_base(
    base: str, start_ms: int, now_ms: int, get: JsonGetter
) -> list[Candle]:
    candles: list[Candle] = []
    cursor = start_ms
    for _ in range(MAX_PAGES):
        query = urlencode(
            {
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "startTime": cursor,
                "limit": PAGE_LIMIT,
            }
        )
        rows = get(f"{base.rstrip('/')}/api/v3/klines?{query}")
        if not isinstance(rows, list):
            raise FetchError("unexpected response shape (expected a JSON array)")
        if not rows:
            return candles
        for row in rows:
            candle = parse_kline(row, now_ms)
            if candle is not None:
                candles.append(candle)
        if len(rows) < PAGE_LIMIT:
            return candles  # a short page is always the last one
        next_cursor = int(rows[-1][0]) + DAY_MS
        if next_cursor <= cursor:
            raise FetchError("pagination did not advance")
        cursor = next_cursor
    raise FetchError(f"more than {MAX_PAGES} pages, refusing to continue")


def fetch_candles(
    bases: Sequence[str],
    start: date,
    *,
    now_ms: int | None = None,
    get: JsonGetter = http_get_json,
) -> list[Candle]:
    """Fetch closed daily candles from ``start`` using the first host that works."""
    now = _utc_ms_now() if now_ms is None else now_ms
    errors: list[str] = []
    for base in bases:
        try:
            return _fetch_from_base(base, _day_to_ms(start), now, get)
        except FetchError as exc:
            log.warning("fetch from %s failed: %s", base, exc)
            errors.append(f"{base}: {exc}")
    raise FetchError("all hosts failed: " + "; ".join(errors))


def read_raw(path: Path) -> dict[date, Candle]:
    if not path.exists():
        return {}
    candles: dict[date, Candle] = {}
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                day = date.fromisoformat(record["date"])
                candles[day] = _validate(
                    Candle(
                        day=day,
                        open=float(record["open"]),
                        high=float(record["high"]),
                        low=float(record["low"]),
                        close=float(record["close"]),
                        volume=float(record["volume"]),
                    )
                )
    except (KeyError, ValueError, OSError) as exc:
        raise FetchError(f"cannot read raw file {path}: {exc}") from exc
    return candles


def write_raw(path: Path, candles: dict[date, Candle]) -> None:
    """Atomically replace the raw file so readers never see a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".raw-", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_FIELDS)
            for day in sorted(candles):
                c = candles[day]
                writer.writerow(
                    [day.isoformat(), c.open, c.high, c.low, c.close, c.volume]
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def run(
    raw_path: Path,
    bases: Sequence[str] = DEFAULT_BASES,
    *,
    now_ms: int | None = None,
    get: JsonGetter = http_get_json,
) -> int:
    """Bring the raw file up to date. Returns how many candles were added or changed."""
    existing = read_raw(raw_path)
    start = max(existing) + timedelta(days=1) if existing else FIRST_DAY
    fetched = fetch_candles(bases, start, now_ms=now_ms, get=get)
    changed = {c.day: c for c in fetched if existing.get(c.day) != c}
    if not changed and existing:
        log.info("raw data already up to date (last day %s)", max(existing))
        return 0
    if not existing and not changed:
        raise FetchError("no candles returned and no existing data")
    existing.update(changed)
    write_raw(raw_path, existing)
    log.info(
        "stored %d new or changed candles, %d total, last day %s",
        len(changed),
        len(existing),
        max(existing),
    )
    return len(changed)


def _bases_from_env() -> list[str]:
    raw = os.environ.get("BINANCE_BASE_URLS", "")
    return [b.strip() for b in raw.split(",") if b.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=Path(os.environ.get("BTC_RAW_PATH", "data/raw/btc_usdt_1d.csv")),
    )
    parser.add_argument(
        "--base-url",
        action="append",
        help="API host, repeatable, tried in order (env: BINANCE_BASE_URLS)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    bases = args.base_url or _bases_from_env() or list(DEFAULT_BASES)
    try:
        run(args.raw_path, bases)
    except FetchError as exc:
        log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
