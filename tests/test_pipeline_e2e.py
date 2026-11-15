"""Fetch, preprocess, train, then serve: the whole data path in one test.

This is the in-process twin of what the Airflow DAG runs as three commands, plus
the API reading the artifact the trainer published.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.app import create_app
from scripts.fake_binance import FakeBinance
from src import etl_fetch, features, train_forecast
from tests.conftest import make_settings


def test_fetch_features_train_then_serve(
    tmp_path: Path,
    fake_binance: FakeBinance,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw" / "btc.csv"
    processed = tmp_path / "processed" / "features.csv"
    models = tmp_path / "models"
    report = tmp_path / "reports" / "last.json"

    # Pin "now" inside the open candle so the run is deterministic.
    monkeypatch.setattr(etl_fetch, "_utc_ms_now", lambda: fake_binance.now_ms)

    assert (
        etl_fetch.main(["--raw-path", str(raw), "--base-url", fake_binance.base_url])
        == 0
    )
    assert features.main(["--raw-path", str(raw), "--out-path", str(processed)]) == 0
    assert (
        train_forecast.main(
            [
                "--features-path",
                str(processed),
                "--model-dir",
                str(models),
                "--report-path",
                str(report),
                "--tolerance",
                "0.5",
            ]
        )
        == 0
    )

    client = create_app(make_settings(models)).test_client()
    assert client.get("/ready").status_code == 200
    body = client.post("/predict").get_json()
    assert len(body["predictions"]) == 7

    open_candle_day = datetime.fromtimestamp(
        fake_binance.klines[-1][0] / 1000, timezone.utc
    ).date()
    last_closed = open_candle_day - timedelta(days=1)
    assert body["base_date"] == last_closed.isoformat()
    assert (
        body["predictions"][0]["date"] == (last_closed + timedelta(days=1)).isoformat()
    )
    assert date.fromisoformat(body["base_date"]) < open_candle_day

    metrics = client.get("/metrics").get_json()["metrics"]
    assert metrics["data_source"] == "binance:BTCUSDT"
    assert metrics["baseline_naive"]["mape"] > 0

    # A second daily run with nothing new must leave the raw data untouched.
    snapshot = raw.read_bytes()
    assert (
        etl_fetch.main(["--raw-path", str(raw), "--base-url", fake_binance.base_url])
        == 0
    )
    assert raw.read_bytes() == snapshot
