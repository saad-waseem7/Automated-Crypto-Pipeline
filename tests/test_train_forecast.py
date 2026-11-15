from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from scripts.seed_artifacts import synthetic_closes
from src.artifacts import load_bundle, read_current
from src.features import FeatureError, Series, compute_features, write_features
from src.forecast_core import ForecastError
from src.train_forecast import main, passes_gate, train, write_report

START = date(2022, 1, 1)


@pytest.fixture(scope="module")
def series_data() -> tuple[list[date], np.ndarray]:
    closes = synthetic_closes(900, seed=5)
    return [START + timedelta(days=i) for i in range(closes.size)], closes


def write_features_file(path: Path, days: list[date], closes: np.ndarray) -> None:
    series = Series(days, closes, np.ones(closes.size), 0)
    write_features(path, series, compute_features(closes))


class TestGate:
    def metrics(self, model: float, naive: float) -> dict[str, object]:
        return {"mape": model, "baseline_naive": {"mape": naive}}

    def test_equal_to_naive_passes(self) -> None:
        assert passes_gate(self.metrics(4.0, 4.0), 0.05)[0]

    def test_within_tolerance_passes_and_beyond_fails(self) -> None:
        assert passes_gate(self.metrics(4.19, 4.0), 0.05)[0]
        assert not passes_gate(self.metrics(4.21, 4.0), 0.05)[0]

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_fails(self, bad: float) -> None:
        assert not passes_gate(self.metrics(bad, 4.0), 0.05)[0]
        assert not passes_gate(self.metrics(4.0, bad), 0.05)[0]


class TestTrain:
    def test_promotes_and_publishes_loadable_artifact(
        self, tmp_path: Path, series_data: tuple[list[date], np.ndarray]
    ) -> None:
        days, closes = series_data
        result = train(days, closes, tmp_path / "models", tolerance=0.5)
        assert result.promoted and result.version is not None
        assert read_current(tmp_path / "models") == result.version
        bundle = load_bundle(tmp_path / "models", result.version)
        assert bundle.last_date == days[-1]
        np.testing.assert_allclose(bundle.recent_closes, closes[-31:])
        assert bundle.metrics["data_source"] == "binance:BTCUSDT"
        assert "gate" in bundle.metrics and bundle.metrics["n_train_closes"] == 900

    def test_rejection_publishes_nothing(
        self, tmp_path: Path, series_data: tuple[list[date], np.ndarray]
    ) -> None:
        days, closes = series_data
        result = train(days, closes, tmp_path / "models", tolerance=-0.99)
        assert not result.promoted and result.version is None
        assert read_current(tmp_path / "models") is None
        assert not (tmp_path / "models" / "versions").exists()

    def test_old_versions_are_pruned_and_current_survives(
        self, tmp_path: Path, series_data: tuple[list[date], np.ndarray]
    ) -> None:
        days, closes = series_data
        models = tmp_path / "models"
        names = [
            train(days, closes, models, tolerance=0.5, keep=2).version for _ in range(4)
        ]
        remaining = sorted(p.name for p in (models / "versions").iterdir())
        assert remaining == sorted(names)[-2:]
        assert read_current(models) == names[-1]

    def test_not_enough_data(
        self, tmp_path: Path, series_data: tuple[list[date], np.ndarray]
    ) -> None:
        days, closes = series_data
        with pytest.raises(ForecastError):
            train(days[:100], closes[:100], tmp_path, eval_origins=60)


class TestReportAndCli:
    def test_report_contents(
        self, tmp_path: Path, series_data: tuple[list[date], np.ndarray]
    ) -> None:
        days, closes = series_data
        result = train(days, closes, tmp_path / "m", tolerance=0.5)
        write_report(tmp_path / "r" / "last.json", result)
        report = json.loads((tmp_path / "r" / "last.json").read_text())
        assert report["status"] == "promoted"
        assert report["version"] == result.version
        assert report["data_last_date"] == days[-1].isoformat()
        assert set(report["metrics"]) == {
            "mae",
            "rmse",
            "mape",
            "baseline_naive",
            "n_origins",
        }

    def test_cli_success_and_rejection_both_exit_zero(
        self,
        tmp_path: Path,
        series_data: tuple[list[date], np.ndarray],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        days, closes = series_data
        features = tmp_path / "features.csv"
        write_features_file(features, days, closes)
        base = [
            "--features-path",
            str(features),
            "--model-dir",
            str(tmp_path / "models"),
            "--report-path",
            str(tmp_path / "report.json"),
        ]
        assert main([*base, "--tolerance", "0.5"]) == 0
        assert "promoted" in capsys.readouterr().out
        assert (
            json.loads((tmp_path / "report.json").read_text())["status"] == "promoted"
        )

        assert main([*base, "--tolerance=-0.99"]) == 0
        assert "rejected" in capsys.readouterr().out
        assert (
            json.loads((tmp_path / "report.json").read_text())["status"] == "rejected"
        )

    def test_cli_unusable_data_exits_2(self, tmp_path: Path) -> None:
        code = main(
            [
                "--features-path",
                str(tmp_path / "none.csv"),
                "--model-dir",
                str(tmp_path / "m"),
                "--report-path",
                str(tmp_path / "r.json"),
            ]
        )
        assert code == 2

    def test_feature_error_type_is_handled(self) -> None:
        assert issubclass(FeatureError, RuntimeError)
