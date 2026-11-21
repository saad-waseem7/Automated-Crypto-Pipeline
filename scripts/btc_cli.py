"""Terminal client for the forecast API. Standard library only.

    python -m scripts.btc_cli status
    python -m scripts.btc_cli predict --days 7
    python -m scripts.btc_cli metrics
    python -m scripts.btc_cli watch --interval 2

The API address comes from ``--url`` or ``BTC_API_URL`` (default
http://127.0.0.1:5000). For Kubernetes use http://btc-forecast.local.

Exit codes: 0 success, 1 the API answered with an error or is not ready,
2 the API could not be reached or the answer was unusable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TextIO
from urllib.parse import urlsplit

DEFAULT_URL = "http://127.0.0.1:5000"
MAX_RESPONSE_BYTES = 1_048_576
SPARK = "▁▂▃▄▅▆▇█"

EXIT_OK, EXIT_API, EXIT_UNREACHABLE = 0, 1, 2


class CliError(Exception):
    """A problem to report on stderr with the given exit code."""

    def __init__(self, message: str, code: int = EXIT_UNREACHABLE) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Reply:
    """An HTTP answer. ``status`` is None when no HTTP response was received."""

    status: int | None
    body: dict[str, Any] | None
    error: str | None = None


class Style:
    """ANSI colors, switched off for pipes, ``NO_COLOR`` and ``TERM=dumb``."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    @classmethod
    def detect(cls, stream: TextIO) -> Style:
        usable = (
            stream.isatty()
            and "NO_COLOR" not in os.environ
            and os.environ.get("TERM") != "dumb"
        )
        return cls(usable)

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)


def normalize_url(raw: str) -> str:
    """Accept only http(s) URLs: urllib would also open file: and ftp: URLs."""
    parts = urlsplit(raw.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise CliError(f"invalid API url {raw!r}: expected http(s)://host[:port]")
    return raw.strip().rstrip("/")


def call(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> Reply:
    data = json.dumps(payload).encode() if payload is not None else None
    # The scheme was restricted to http(s) by normalize_url.
    request = urllib.request.Request(
        base_url + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _decode(response.status, response.read(MAX_RESPONSE_BYTES + 1))
    except urllib.error.HTTPError as exc:
        return _decode(exc.code, exc.read(MAX_RESPONSE_BYTES + 1))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return Reply(None, None, f"cannot reach {base_url}: {reason}")


def _decode(status: int, raw: bytes) -> Reply:
    if len(raw) > MAX_RESPONSE_BYTES:
        return Reply(status, None, "response too large")
    try:
        body = json.loads(raw)
    except ValueError:
        return Reply(status, None, f"HTTP {status} with a non-JSON body")
    if not isinstance(body, dict):
        return Reply(status, None, f"HTTP {status} with an unexpected JSON shape")
    return Reply(status, body)


def api_error_text(reply: Reply) -> str:
    if reply.error:
        return reply.error
    error = (reply.body or {}).get("error")
    if isinstance(error, dict) and "message" in error:
        return f"HTTP {reply.status}: {error['message']}"
    return f"HTTP {reply.status}"


def exit_code_for(reply: Reply) -> int:
    return EXIT_UNREACHABLE if reply.status is None else EXIT_API


def format_uptime(seconds: float) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return SPARK[3] * len(values)
    top = len(SPARK) - 1
    return "".join(SPARK[round((v - low) / (high - low) * top)] for v in values)


def signed(value: float, decimals: int = 2) -> str:
    return f"{value:+,.{decimals}f}"


def render_status(
    url: str, health: Reply, ready: Reply, style: Style
) -> tuple[str, int]:
    """Return the status text and the exit code."""
    lines = [f"{'API':<11}{url}"]
    if health.status is None:
        lines.append(f"{'Liveness':<11}{style.red('unreachable')}  {health.error}")
        return "\n".join(lines), EXIT_UNREACHABLE
    body = health.body or {}
    uptime = body.get("uptime_seconds")
    suffix = f"  (up {format_uptime(float(uptime))})" if uptime is not None else ""
    lines.append(f"{'Liveness':<11}{style.green('ok')}{suffix}")

    if ready.status == 200:
        lines.append(f"{'Readiness':<11}{style.green('ready')}")
        code = EXIT_OK
    else:
        lines.append(
            f"{'Readiness':<11}{style.yellow('NOT READY')}  {api_error_text(ready)}"
        )
        code = EXIT_API
    version = body.get("model_version") or "none loaded"
    lines.append(f"{'Model':<11}{version}  (stage: {body.get('model_stage', '?')})")
    return "\n".join(lines), code


def render_predict(body: dict[str, Any], style: Style) -> str:
    base = float(body["base_close"])
    rows = [(str(p["date"]), float(p["predicted_close"])) for p in body["predictions"]]
    lines = [
        f"{style.bold('Model')} {body['model_version']}   "
        f"{style.bold('Last close')} {base:,.2f} on {body['base_date']}",
        "",
        f"{'Date':<12}{'Forecast':>14}{'vs last':>12}{'%':>9}",
    ]
    for day, price in rows:
        delta = price - base
        pct = delta / base * 100
        text = f"{day:<12}{price:>14,.2f}{signed(delta):>12}{signed(pct) + '%':>9}"
        lines.append(style.green(text) if delta >= 0 else style.red(text))
    lines += [
        "",
        f"Trend  {sparkline([base] + [price for _, price in rows])}  "
        + style.dim("(last close, then forecast)"),
        style.dim(
            "Not trading advice: the model performs about like 'tomorrow equals today'."
        ),
    ]
    return "\n".join(lines)


def render_metrics(body: dict[str, Any], style: Style) -> str:
    metrics = body["metrics"]
    naive = metrics["baseline_naive"]
    lines = [
        f"{style.bold('Model')} {body['model_version']}   "
        f"trained {body.get('trained_at', '?')}",
        f"Data {metrics.get('data_source', '?')}, {metrics.get('n_train_closes', '?')} "
        f"closes, backtest over {metrics.get('n_origins', '?')} origins",
        "",
        f"{'':<8}{'model':>12}{'naive':>12}{'diff':>10}",
    ]
    for key, label in (("mae", "MAE"), ("rmse", "RMSE"), ("mape", "MAPE %")):
        model_value, naive_value = float(metrics[key]), float(naive[key])
        relative = (model_value / naive_value - 1) * 100 if naive_value else 0.0
        diff = f"{relative:+.1f}%"
        diff = style.red(diff) if relative > 0 else style.green(diff)
        lines.append(f"{label:<8}{model_value:>12,.3f}{naive_value:>12,.3f}{diff:>10}")
    lines += ["", f"Gate: {metrics.get('gate', 'n/a')}"]
    lines.append(
        style.dim("Positive diff means worse than the naive 'tomorrow equals today'.")
    )
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace, out: TextIO, style: Style) -> int:
    health = call(args.url, "/health", timeout=args.timeout)
    ready = call(args.url, "/ready", timeout=args.timeout)
    text, code = render_status(args.url, health, ready, style)
    print(text, file=out)
    return code


def _fetch_and_render(
    args: argparse.Namespace,
    out: TextIO,
    style: Style,
    request: Callable[[], Reply],
    render: Callable[[dict[str, Any], Style], str],
) -> int:
    reply = request()
    if reply.status != 200 or reply.body is None:
        raise CliError(api_error_text(reply), exit_code_for(reply))
    try:
        print(render(reply.body, style), file=out)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise CliError(f"unexpected response from the API: {exc!r}") from exc
    return EXIT_OK


def cmd_predict(args: argparse.Namespace, out: TextIO, style: Style) -> int:
    return _fetch_and_render(
        args,
        out,
        style,
        lambda: call(
            args.url,
            "/predict",
            method="POST",
            payload={"forecast_days": args.days},
            timeout=args.timeout,
        ),
        render_predict,
    )


def cmd_metrics(args: argparse.Namespace, out: TextIO, style: Style) -> int:
    return _fetch_and_render(
        args,
        out,
        style,
        lambda: call(args.url, "/metrics", timeout=args.timeout),
        render_metrics,
    )


def cmd_watch(
    args: argparse.Namespace,
    out: TextIO,
    style: Style,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = datetime.now,
) -> int:
    """Print one line per poll. Marks the moment the served model version changes."""
    previous: str | None = None
    tick = 0
    try:
        while args.count == 0 or tick < args.count:
            if tick:
                sleep(args.interval)
            tick += 1
            ready = call(args.url, "/ready", timeout=args.timeout)
            stamp = now().strftime("%H:%M:%S")
            if ready.status is None:
                print(f"{stamp}  {style.red('unreachable')}  {ready.error}", file=out)
            elif ready.status == 200 and ready.body:
                version = str(ready.body.get("model_version"))
                note = ""
                if previous is not None and version != previous:
                    note = style.yellow("  <- model version changed")
                previous = version
                print(f"{stamp}  {style.green('ready')}      {version}{note}", file=out)
            else:
                print(
                    f"{stamp}  {style.yellow('not ready')}  {api_error_text(ready)}",
                    file=out,
                )
            out.flush()
    except KeyboardInterrupt:
        pass
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="btc_cli", description="Terminal client for the BTC forecast API."
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("BTC_API_URL", DEFAULT_URL),
        help="API base URL (env BTC_API_URL, default %(default)s)",
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0, help="seconds per request"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="liveness, readiness and served model version")
    predict = sub.add_parser("predict", help="7-day forecast table")
    predict.add_argument("--days", type=int, default=7, choices=range(1, 8))
    sub.add_parser("metrics", help="model versus naive baseline and the gate result")
    watch = sub.add_parser("watch", help="poll readiness and flag model changes")
    watch.add_argument("--interval", type=float, default=2.0, help="seconds")
    watch.add_argument(
        "--count", type=int, default=0, help="stop after N polls (0=never)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        print("error: --timeout must be positive", file=sys.stderr)
        return EXIT_UNREACHABLE
    if args.command == "watch" and (args.interval <= 0 or args.count < 0):
        print(
            "error: --interval must be positive and --count non-negative",
            file=sys.stderr,
        )
        return EXIT_UNREACHABLE
    style = Style.detect(sys.stdout)
    handlers: dict[str, Callable[[argparse.Namespace, TextIO, Style], int]] = {
        "status": cmd_status,
        "predict": cmd_predict,
        "metrics": cmd_metrics,
        "watch": cmd_watch,
    }
    try:
        args.url = normalize_url(args.url)
        return handlers[args.command](args, sys.stdout, style)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code
    except BrokenPipeError:
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
