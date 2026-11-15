from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from scripts.fake_binance import FakeBinance
from src.etl_fetch import Candle, write_raw
from src.features import (
    FEATURE_FIELDS,
    FeatureError,
    compute_features,
    fill_gaps,
    main,
    read_closes,
    run,
)
from src.forecast_core import HORIZON

D0 = date(2024, 1, 1)


def days(n: int, skip: set[int] | None = None) -> list[date]:
    return [D0 + timedelta(days=i) for i in range(n) if i not in (skip or set())]


class TestFillGaps:
    def test_no_gaps_is_identity(self) -> None:
        series = fill_gaps(days(5), np.arange(1.0, 6.0), np.ones(5))
        assert series.days == days(5)
        assert series.filled == 0

    def test_short_gap_is_forward_filled_with_zero_volume(self) -> None:
        d = days(6, skip={2, 3})
        series = fill_gaps(d, np.array([1.0, 2.0, 5.0, 6.0]), np.array([9.0] * 4))
        assert series.days == days(6)
        assert series.close.tolist() == [1.0, 2.0, 2.0, 2.0, 5.0, 6.0]
        assert series.volume.tolist() == [9.0, 9.0, 0.0, 0.0, 9.0, 9.0]
        assert series.filled == 2

    def test_long_gap_is_an_error(self) -> None:
        d = days(10, skip={2, 3, 4, 5})
        with pytest.raises(FeatureError, match="gap of 4"):
            fill_gaps(d, np.ones(6), np.ones(6))

    def test_gap_at_the_limit_is_allowed(self) -> None:
        d = days(8, skip={2, 3, 4})
        assert fill_gaps(d, np.ones(5), np.ones(5)).filled == 3

    def test_empty_and_unsorted_are_errors(self) -> None:
        with pytest.raises(FeatureError):
            fill_gaps([], np.array([]), np.array([]))
        with pytest.raises(FeatureError, match="increasing"):
            fill_gaps([D0 + timedelta(days=1), D0], np.ones(2), np.ones(2))


class TestComputeFeatures:
    @pytest.fixture
    def close(self) -> np.ndarray:
        rng = np.random.default_rng(3)
        return np.asarray(100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 80))))

    def test_log_return(self, close: np.ndarray) -> None:
        f = compute_features(close)
        assert np.isnan(f["log_return"][0])
        np.testing.assert_allclose(f["log_return"][1:], np.diff(np.log(close)))

    def test_moving_averages(self, close: np.ndarray) -> None:
        f = compute_features(close)
        assert np.isnan(f["ma_7"][:6]).all() and not np.isnan(f["ma_7"][6])
        assert np.isnan(f["ma_30"][:29]).all() and not np.isnan(f["ma_30"][29])
        np.testing.assert_allclose(f["ma_7"][6], close[:7].mean())
        np.testing.assert_allclose(f["ma_30"][40], close[11:41].mean())

    def test_volatility_is_sample_std_of_30_returns(self, close: np.ndarray) -> None:
        f = compute_features(close)
        returns = np.diff(np.log(close))
        assert np.isnan(f["vol_30"][:30]).all()
        np.testing.assert_allclose(f["vol_30"][30], returns[0:30].std(ddof=1))
        np.testing.assert_allclose(f["vol_30"][-1], returns[-30:].std(ddof=1))

    def test_targets_match_definition(self, close: np.ndarray) -> None:
        f = compute_features(close)
        for h in range(1, HORIZON + 1):
            target = f[f"y_h{h}"]
            assert np.isnan(target[-h:]).all()
            assert not np.isnan(target[:-h]).any()
            np.testing.assert_allclose(target[10], np.log(close[10 + h] / close[10]))

    def test_short_series_does_not_crash(self) -> None:
        f = compute_features(np.array([1.0, 2.0, 3.0]))
        assert np.isnan(f["ma_7"]).all()
        assert np.isnan(f["y_h5"]).all()


class TestPipeline:
    def test_raw_to_features_roundtrip(self, tmp_path: Path) -> None:
        raw, out = tmp_path / "raw.csv", tmp_path / "p" / "features.csv"
        candles = {
            d: Candle(d, 100.0 + i, 120.0 + i, 90.0 + i, 100.0 + i, 5.0)
            for i, d in enumerate(days(60, skip={20}))
        }
        write_raw(raw, candles)
        series = run(raw, out)
        assert series.filled == 1
        with open(out, newline="") as handle:
            reader = csv.DictReader(handle)
            assert tuple(reader.fieldnames or ()) == FEATURE_FIELDS
            rows = list(reader)
        assert len(rows) == 60
        assert rows[0]["log_return"] == "" and rows[0]["ma_7"] == ""
        assert rows[-1]["y_h1"] == ""
        loaded_days, closes = read_closes(out)
        assert loaded_days == days(60)
        assert closes[20] == closes[19]  # forward-filled day
        assert out.stat().st_mode & 0o777 == 0o644

    def test_run_without_raw_data(self, tmp_path: Path) -> None:
        with pytest.raises(FeatureError, match="no raw data"):
            run(tmp_path / "missing.csv", tmp_path / "out.csv")

    def test_read_closes_rejects_gaps_and_garbage(self, tmp_path: Path) -> None:
        path = tmp_path / "f.csv"
        path.write_text("date,close\n2024-01-01,1\n2024-01-03,2\n")
        with pytest.raises(FeatureError, match="gap"):
            read_closes(path)
        path.write_text("date,close\n")
        with pytest.raises(FeatureError, match="empty"):
            read_closes(path)
        path.write_text("date,close\nnot-a-date,1\n")
        with pytest.raises(FeatureError):
            read_closes(path)
        with pytest.raises(FeatureError):
            read_closes(tmp_path / "absent.csv")

    def test_cli(self, tmp_path: Path, fake_binance: FakeBinance) -> None:
        from src.etl_fetch import run as fetch

        raw = tmp_path / "raw.csv"
        fetch(raw, [fake_binance.base_url], now_ms=fake_binance.now_ms)
        out = tmp_path / "features.csv"
        assert main(["--raw-path", str(raw), "--out-path", str(out)]) == 0
        assert out.exists()
        assert (
            main(["--raw-path", str(tmp_path / "no.csv"), "--out-path", str(out)]) == 2
        )
