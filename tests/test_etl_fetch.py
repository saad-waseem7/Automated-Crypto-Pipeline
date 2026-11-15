from __future__ import annotations

import csv
import functools
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.fake_binance import DAY_MS, FakeBinance
from src import etl_fetch
from src.etl_fetch import (
    Candle,
    FetchError,
    fetch_candles,
    http_get_json,
    main,
    parse_kline,
    read_raw,
    run,
    write_raw,
)

NO_SLEEP = functools.partial(http_get_json, sleep=lambda _: None)
FAR_FUTURE = 10**15


def kline(open_ms: int = 1_700_006_400_000, **overrides: Any) -> list[Any]:
    row: list[Any] = [
        open_ms,
        "100.0",
        "110.0",
        "90.0",
        "105.0",
        "12.5",
        open_ms + DAY_MS - 1,
        "0",
        1,
        "0",
        "0",
        "0",
    ]
    for index, value in overrides.items():
        row[int(index.lstrip("i"))] = value
    return row


class TestParseKline:
    def test_closed_candle(self) -> None:
        candle = parse_kline(kline(), FAR_FUTURE)
        assert candle is not None
        assert candle.close == 105.0
        assert candle.day == date(2023, 11, 15)

    def test_open_candle_is_skipped(self) -> None:
        row = kline()
        assert parse_kline(row, int(row[6])) is None
        assert parse_kline(row, int(row[6]) + 1) is not None

    @pytest.mark.parametrize(
        "row",
        [
            [],
            [1, 2],
            kline(i1="abc"),
            kline(i4=None),
            kline(i0="x"),
            kline(i2="50.0"),  # high below open and close
            kline(i3="200.0"),  # low above open and close
            kline(i4="-1"),
            kline(i4="nan"),
            kline(i5="-3"),
            kline(i0=1_700_006_400_001),  # not aligned to a UTC day
        ],
    )
    def test_corrupt_rows_rejected(self, row: list[Any]) -> None:
        with pytest.raises(FetchError):
            parse_kline(row, FAR_FUTURE)


class TestFetching:
    def test_paginates_and_drops_open_candle(self) -> None:
        with FakeBinance(n_days=2300) as server:
            candles = fetch_candles(
                [server.base_url], date(2017, 8, 17), now_ms=server.now_ms
            )
            assert server.requests == 3
        assert len(candles) == 2299
        today = datetime.fromtimestamp(server.klines[-1][0] / 1000, timezone.utc).date()
        assert candles[-1].day == today - timedelta(days=1)
        assert today not in {c.day for c in candles}
        days = [c.day for c in candles]
        assert days == sorted(set(days))

    def test_retries_transient_errors(self) -> None:
        with FakeBinance(n_days=40, fail_first=2) as server:
            candles = fetch_candles(
                [server.base_url], date(2017, 8, 17), now_ms=server.now_ms, get=NO_SLEEP
            )
            assert server.requests == 3
        assert len(candles) == 39

    def test_gives_up_after_repeated_failures(self) -> None:
        with FakeBinance(n_days=40, fail_first=99) as server:
            with pytest.raises(FetchError, match="all hosts failed"):
                fetch_candles(
                    [server.base_url],
                    date(2017, 8, 17),
                    now_ms=server.now_ms,
                    get=NO_SLEEP,
                )

    def test_falls_back_to_second_host_when_first_is_geo_blocked(self) -> None:
        with (
            FakeBinance(n_days=40, status=451) as blocked,
            FakeBinance(n_days=40) as ok,
        ):
            candles = fetch_candles(
                [blocked.base_url, ok.base_url],
                date(2017, 8, 17),
                now_ms=ok.now_ms,
            )
            assert blocked.requests == 1  # 451 is not retried
        assert len(candles) == 39

    def test_all_hosts_failing_reports_each(self) -> None:
        with FakeBinance(status=451) as a, FakeBinance(status=451) as b:
            with pytest.raises(FetchError) as info:
                fetch_candles([a.base_url, b.base_url], date(2017, 8, 17))
        assert "451" in str(info.value)

    def test_unreachable_host_raises(self) -> None:
        with pytest.raises(FetchError):
            fetch_candles(
                ["http://127.0.0.1:1"],
                date(2017, 8, 17),
                get=functools.partial(http_get_json, retries=1, sleep=lambda _: None),
            )

    def test_non_list_response_rejected(self) -> None:
        with pytest.raises(FetchError, match="array"):
            fetch_candles(["http://x"], date(2017, 8, 17), get=lambda _url: {"a": 1})

    def test_runaway_pagination_is_stopped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(etl_fetch, "MAX_PAGES", 3)
        monkeypatch.setattr(etl_fetch, "PAGE_LIMIT", 2)
        counter = {"open": 1_700_006_400_000}

        def endless(_url: str) -> list[list[Any]]:
            rows = []
            for _ in range(2):
                rows.append(kline(counter["open"]))
                counter["open"] += DAY_MS
            return rows

        with pytest.raises(FetchError, match="pages"):
            etl_fetch._fetch_from_base("http://x", 0, FAR_FUTURE, endless)

    def test_stalled_pagination_is_detected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(etl_fetch, "PAGE_LIMIT", 1)
        row = kline(1_700_006_400_000)
        with pytest.raises(FetchError, match="advance"):
            etl_fetch._fetch_from_base(
                "http://x", 1_700_006_400_000 + 5 * DAY_MS, FAR_FUTURE, lambda _u: [row]
            )


class TestRawStorage:
    def test_run_is_idempotent_and_incremental(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw" / "btc.csv"
        with FakeBinance(n_days=200) as server:
            midpoint = int(server.klines[100][0]) + DAY_MS  # candles 0..100 closed
            first = run(raw, [server.base_url], now_ms=midpoint)
            assert first == 101
            assert len(read_raw(raw)) == 101

            second = run(raw, [server.base_url], now_ms=server.now_ms)
            assert second == 98  # candles 101..198, the open one excluded
            snapshot = raw.read_bytes()

            third = run(raw, [server.base_url], now_ms=server.now_ms)
            assert third == 0
            assert raw.read_bytes() == snapshot
        assert len(read_raw(raw)) == 199

    def test_revised_candle_overwrites_stored_one(self, tmp_path: Path) -> None:
        raw = tmp_path / "btc.csv"
        day = date(2023, 11, 15)
        write_raw(raw, {day: Candle(day, 100.0, 110.0, 90.0, 105.0, 12.5)})
        revised = kline(i4="120.0", i2="130.0")  # same day, close revised to 120
        assert run(raw, ["http://x"], now_ms=FAR_FUTURE, get=lambda _u: [revised]) == 1
        assert read_raw(raw)[day].close == 120.0

    def test_empty_response_without_existing_data_is_an_error(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(FetchError, match="no candles"):
            run(tmp_path / "btc.csv", ["http://x"], get=lambda _u: [])

    def test_roundtrip_and_permissions(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "btc.csv"
        data = {date(2024, 1, 1): Candle(date(2024, 1, 1), 1.0, 2.0, 0.5, 1.5, 3.0)}
        write_raw(path, data)
        assert read_raw(path) == data
        assert path.stat().st_mode & 0o777 == 0o644
        assert [p.name for p in path.parent.iterdir()] == ["btc.csv"]

    def test_failed_write_keeps_original_and_cleans_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "btc.csv"
        good = {date(2024, 1, 1): Candle(date(2024, 1, 1), 1.0, 2.0, 0.5, 1.5, 3.0)}
        write_raw(path, good)
        original = path.read_bytes()

        def boom(*_: object, **__: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(etl_fetch.os, "replace", boom)
        with pytest.raises(OSError):
            write_raw(path, {})
        assert path.read_bytes() == original
        assert [p.name for p in tmp_path.iterdir()] == ["btc.csv"]

    def test_corrupt_raw_file_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "btc.csv"
        path.write_text("date,open,high,low,close,volume\n2024-01-01,a,b,c,d,e\n")
        with pytest.raises(FetchError):
            read_raw(path)
        path.write_text("wrong,header\n1,2\n")
        with pytest.raises(FetchError):
            read_raw(path)

    def test_csv_has_expected_header(self, tmp_path: Path) -> None:
        with FakeBinance(n_days=10) as server:
            run(tmp_path / "btc.csv", [server.base_url], now_ms=server.now_ms)
        with open(tmp_path / "btc.csv", newline="") as handle:
            assert next(csv.reader(handle)) == list(etl_fetch.CSV_FIELDS)


class TestCli:
    def test_success_uses_env_hosts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with FakeBinance(n_days=30) as server:
            monkeypatch.setenv("BINANCE_BASE_URLS", server.base_url)
            assert main(["--raw-path", str(tmp_path / "btc.csv")]) == 0
        assert len(read_raw(tmp_path / "btc.csv")) == 29

    def test_flag_overrides_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BINANCE_BASE_URLS", "http://127.0.0.1:1")
        with FakeBinance(n_days=30) as server:
            code = main(
                ["--raw-path", str(tmp_path / "btc.csv"), "--base-url", server.base_url]
            )
        assert code == 0

    def test_failure_returns_exit_code_2(self, tmp_path: Path) -> None:
        with FakeBinance(status=451) as server:
            code = main(
                ["--raw-path", str(tmp_path / "btc.csv"), "--base-url", server.base_url]
            )
        assert code == 2
        assert not (tmp_path / "btc.csv").exists()
