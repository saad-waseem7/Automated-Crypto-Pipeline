"""A tiny fake of the Binance klines endpoint, serving SYNTHETIC prices.

Used by unit tests, CI and offline development so the real HTTP code path in
``src.etl_fetch`` runs without internet access. It mimics the real wire format:
every number is a string and the last candle (today, UTC) is still open.

Usage: python -m scripts.fake_binance --port 8765 --days 1500
       BINANCE_BASE_URLS=http://127.0.0.1:8765 python -m src.etl_fetch
"""

from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

DAY_MS = 86_400_000
MAX_LIMIT = 1000


def _build_klines(n_days: int, seed: int) -> list[list[Any]]:
    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    first_open_ms = int((today - timedelta(days=n_days - 1)).timestamp() * 1000)
    rng = np.random.default_rng(seed)
    closes = 30_000.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n_days)))
    rows: list[list[Any]] = []
    previous = float(closes[0])
    for i, close in enumerate(closes):
        open_ms = first_open_ms + i * DAY_MS
        close = float(close)
        high = max(previous, close) * 1.01
        low = min(previous, close) * 0.99
        rows.append(
            [
                open_ms,
                f"{previous:.8f}",
                f"{high:.8f}",
                f"{low:.8f}",
                f"{close:.8f}",
                f"{1000 + i % 50:.8f}",
                open_ms + DAY_MS - 1,
                "0.00000000",
                100,
                "0.00000000",
                "0.00000000",
                "0",
            ]
        )
        previous = close
    return rows


class FakeBinance:
    """Context manager running the fake server on 127.0.0.1 and a free port.

    ``fail_first`` makes the first N requests answer 503, and ``status`` makes
    every request answer with that HTTP status, to exercise retry and fallback.
    """

    def __init__(
        self,
        n_days: int = 1500,
        seed: int = 11,
        *,
        fail_first: int = 0,
        status: int | None = None,
        port: int = 0,
    ) -> None:
        self.klines = _build_klines(n_days, seed)
        self.fail_first = fail_first
        self.status = status
        self.requests = 0
        self._port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def now_ms(self) -> int:
        """A clock reading inside the open (last) candle, for deterministic tests."""
        return int(self.klines[-1][0]) + DAY_MS // 2

    @property
    def base_url(self) -> str:
        assert self._server is not None, "server not started"
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 (http.server API)
                outer.requests += 1
                if outer.status is not None:
                    self.send_error(outer.status)
                    return
                if outer.requests <= outer.fail_first:
                    self.send_error(503)
                    return
                parsed = urlparse(self.path)
                if parsed.path != "/api/v3/klines":
                    self.send_error(404)
                    return
                query = parse_qs(parsed.query)
                start = int(query.get("startTime", ["0"])[0])
                limit = min(int(query.get("limit", ["500"])[0]), MAX_LIMIT)
                rows = [r for r in outer.klines if r[0] >= start][:limit]
                body = json.dumps(rows).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return

        return Handler

    def start(self) -> FakeBinance:
        self._server = ThreadingHTTPServer(
            ("127.0.0.1", self._port), self._make_handler()
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.02},
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> FakeBinance:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--days", type=int, default=1500)
    args = parser.parse_args(argv)
    server = FakeBinance(args.days, port=args.port).start()
    print(f"fake Binance listening on {server.base_url}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
