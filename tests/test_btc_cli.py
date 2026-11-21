from __future__ import annotations

import io
import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from werkzeug.serving import make_server

from app.app import create_app
from scripts import btc_cli
from scripts.btc_cli import (
    CliError,
    Style,
    build_parser,
    call,
    cmd_watch,
    format_uptime,
    main,
    normalize_url,
    sparkline,
)
from tests.conftest import make_settings

Publish = Callable[..., str]


@contextmanager
def serve_app(model_dir: Path) -> Iterator[str]:
    server = make_server("127.0.0.1", 0, create_app(make_settings(model_dir)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@contextmanager
def canned_server(status: int, body: bytes) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self) -> None:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = _reply  # noqa: N815

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def api(tmp_path: Path, publish: Publish) -> Iterator[str]:
    publish(tmp_path)
    with serve_app(tmp_path) as url:
        yield url


@pytest.fixture
def empty_api(tmp_path: Path) -> Iterator[str]:
    with serve_app(tmp_path / "empty") as url:
        yield url


class TestHelpers:
    @pytest.mark.parametrize(
        "raw", ["", "file:///etc/passwd", "ftp://host", "localhost:5000", "http://"]
    )
    def test_only_http_urls_are_accepted(self, raw: str) -> None:
        with pytest.raises(CliError):
            normalize_url(raw)

    def test_trailing_slash_is_removed(self) -> None:
        assert normalize_url(" http://h:1/ ") == "http://h:1"

    @pytest.mark.parametrize(
        "seconds, text",
        [(5, "5s"), (65, "1m 05s"), (3725, "1h 02m 05s"), (0.9, "0s")],
    )
    def test_format_uptime(self, seconds: float, text: str) -> None:
        assert format_uptime(seconds) == text

    def test_sparkline(self) -> None:
        assert sparkline([]) == ""
        assert sparkline([5.0, 5.0, 5.0]) == "▄▄▄"
        line = sparkline([1.0, 2.0, 3.0, 4.0])
        assert line[0] == "▁" and line[-1] == "█" and len(line) == 4

    def test_colors_only_when_enabled(self) -> None:
        assert Style(False).green("x") == "x"
        assert Style(True).green("x") == "\033[32mx\033[0m"

    def test_no_color_and_non_tty_disable_styling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Tty(io.StringIO):
            def isatty(self) -> bool:
                return True

        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm")
        assert Style.detect(Tty()).enabled
        assert not Style.detect(io.StringIO()).enabled
        monkeypatch.setenv("NO_COLOR", "1")
        assert not Style.detect(Tty()).enabled


class TestCall:
    def test_unreachable_has_no_status(self) -> None:
        reply = call("http://127.0.0.1:1", "/health", timeout=0.5)
        assert reply.status is None and "cannot reach" in (reply.error or "")

    def test_non_json_body(self) -> None:
        with canned_server(200, b"<html>hi</html>") as url:
            reply = call(url, "/health")
        assert reply.status == 200 and "non-JSON" in (reply.error or "")

    def test_unexpected_json_shape(self) -> None:
        with canned_server(200, b"[1, 2]") as url:
            assert "unexpected JSON" in (call(url, "/health").error or "")

    def test_oversized_response_is_refused(self) -> None:
        with canned_server(200, b"x" * (btc_cli.MAX_RESPONSE_BYTES + 10)) as url:
            assert call(url, "/health").error == "response too large"

    def test_error_status_keeps_the_json_body(self) -> None:
        body = json.dumps({"error": {"code": "x", "message": "boom"}}).encode()
        with canned_server(503, body) as url:
            reply = call(url, "/ready")
        assert reply.status == 503 and btc_cli.api_error_text(reply) == "HTTP 503: boom"


class TestCommands:
    def test_status_ready(self, api: str, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--url", api, "status"]) == 0
        out = capsys.readouterr().out
        assert "ready" in out and "v1" in out and "\033" not in out

    def test_status_not_ready_exits_1(
        self, empty_api: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--url", empty_api, "status"]) == 1
        out = capsys.readouterr().out
        assert "NOT READY" in out and "ok" in out  # alive but not ready

    def test_status_unreachable_exits_2(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--url", "http://127.0.0.1:1", "--timeout", "0.5", "status"]) == 2
        assert "unreachable" in capsys.readouterr().out

    def test_predict_table(self, api: str, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--url", api, "predict"]) == 0
        out = capsys.readouterr().out
        assert "Last close" in out and "Trend" in out
        assert sum(line[:2] == "20" for line in out.splitlines()) == 7

    def test_predict_days_option(
        self, api: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--url", api, "predict", "--days", "3"]) == 0
        out = capsys.readouterr().out
        assert sum(line[:2] == "20" for line in out.splitlines()) == 3

    def test_predict_without_model_exits_1(
        self, empty_api: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--url", empty_api, "predict"]) == 1
        assert "no model is loaded" in capsys.readouterr().err

    def test_predict_with_malformed_answer_exits_2(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with canned_server(200, b'{"model_version": "x"}') as url:
            assert main(["--url", url, "predict"]) == 2
        assert "unexpected response" in capsys.readouterr().err

    def test_metrics(self, api: str, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--url", api, "metrics"]) == 0
        out = capsys.readouterr().out
        assert "MAPE" in out and "naive" in out and "Gate:" in out

    def test_invalid_url_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--url", "file:///etc/passwd", "status"]) == 2
        assert "invalid API url" in capsys.readouterr().err

    def test_url_comes_from_the_environment(
        self,
        api: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("BTC_API_URL", api)
        assert main(["status"]) == 0
        assert api in capsys.readouterr().out

    @pytest.mark.parametrize(
        "argv",
        [
            ["--timeout", "0", "status"],
            ["watch", "--interval", "0"],
            ["watch", "--count", "-1"],
        ],
    )
    def test_bad_numbers_are_rejected(
        self, argv: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(argv) == 2
        assert "error" in capsys.readouterr().err

    def test_days_out_of_range_is_a_usage_error(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["predict", "--days", "9"])


class TestWatch:
    def run_watch(
        self, url: str, count: int, sleep: Callable[[float], None]
    ) -> list[str]:
        args = build_parser().parse_args(
            ["--url", url, "watch", "--count", str(count), "--interval", "0.01"]
        )
        out = io.StringIO()
        cmd_watch(
            args, out, Style(False), sleep=sleep, now=lambda: datetime(2026, 1, 1)
        )
        return out.getvalue().splitlines()

    def test_flags_a_model_version_change(
        self, tmp_path: Path, publish: Publish
    ) -> None:
        publish(tmp_path, "v1")
        with serve_app(tmp_path) as url:
            published = {"done": False}

            def sleep(_: float) -> None:
                if not published["done"]:
                    publish(tmp_path, "v2")
                    published["done"] = True

            lines = self.run_watch(url, 3, sleep)
        assert len(lines) == 3
        assert "v1" in lines[0] and "changed" not in lines[0]
        assert "v2" in lines[1] and "model version changed" in lines[1]
        assert "changed" not in lines[2]

    def test_reports_unreachable_and_not_ready(self, empty_api: str) -> None:
        assert "not ready" in self.run_watch(empty_api, 1, lambda _: None)[0]
        down = self.run_watch("http://127.0.0.1:1", 1, lambda _: None)
        assert "unreachable" in down[0]

    def test_ctrl_c_stops_cleanly(self, api: str) -> None:
        def interrupt(_: float) -> None:
            raise KeyboardInterrupt

        assert len(self.run_watch(api, 0, interrupt)) == 1
