from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from flask import Flask
from flask.testing import FlaskClient

from app.app import create_app
from app.config import Settings
from scripts.fake_binance import FakeBinance
from scripts.seed_artifacts import synthetic_closes
from src.artifacts import write_version
from src.forecast_core import FloatArray, evaluate_holdout, fit_ridge

TOKEN = "test-token-0123456789"
LAST_DATE = date(2026, 1, 31)


@pytest.fixture(scope="session")
def closes() -> FloatArray:
    return synthetic_closes(600, seed=7)


@pytest.fixture
def publish(closes: FloatArray) -> Callable[..., str]:
    """Write a model version into a directory and return its name."""

    def _publish(model_dir: Path, version: str = "v1", promote: bool = True) -> str:
        model = fit_ridge(closes)
        metrics = evaluate_holdout(model, closes, 20)
        return write_version(
            model_dir,
            model,
            closes,
            LAST_DATE,
            metrics,
            version=version,
            promote=promote,
        )

    return _publish


def make_settings(model_dir: Path, **overrides: object) -> Settings:
    base = {
        "model_dir": model_dir,
        "poll_seconds": 0,
        "reload_token": TOKEN,
        "max_history": 1000,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture
def app_with_model(tmp_path: Path, publish: Callable[..., str]) -> Flask:
    publish(tmp_path)
    return create_app(make_settings(tmp_path))


@pytest.fixture
def client(app_with_model: Flask) -> FlaskClient:
    return app_with_model.test_client()


@pytest.fixture
def empty_client(tmp_path: Path) -> FlaskClient:
    return create_app(make_settings(tmp_path / "empty")).test_client()


@pytest.fixture
def history(closes: FloatArray) -> list[float]:
    return [float(v) for v in np.asarray(closes)[-40:]]


@pytest.fixture
def fake_binance() -> Iterator[FakeBinance]:
    with FakeBinance(n_days=1200) as server:
        yield server
