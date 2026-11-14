from __future__ import annotations

import numpy as np
import pytest

from src.forecast_core import (
    HORIZON,
    N_LAGS,
    ForecastError,
    RidgeForecaster,
    evaluate_holdout,
    fit_ridge,
    validate_closes,
)


def test_fit_shapes_and_determinism(closes: np.ndarray) -> None:
    a = fit_ridge(closes)
    b = fit_ridge(closes)
    assert a.coef.shape == (N_LAGS, HORIZON)
    assert a.intercept.shape == (HORIZON,)
    np.testing.assert_array_equal(a.coef, b.coef)


def test_predict_length_and_positive(closes: np.ndarray) -> None:
    prices = fit_ridge(closes).predict(closes)
    assert prices.shape == (HORIZON,)
    assert (prices > 0).all()


def test_predict_uses_only_tail(closes: np.ndarray) -> None:
    model = fit_ridge(closes)
    np.testing.assert_allclose(
        model.predict(closes), model.predict(closes[-model.min_history :])
    )


def test_forecast_stays_near_last_close(closes: np.ndarray) -> None:
    model = fit_ridge(closes)
    ratio = model.predict(closes) / closes[-1]
    assert ((ratio > 0.5) & (ratio < 2.0)).all()


@pytest.mark.parametrize(
    "bad",
    [[], [1.0] * 5, [float("nan")] * 200, [float("inf")] * 200, [-1.0] * 200],
)
def test_validate_rejects_bad_closes(bad: list[float]) -> None:
    with pytest.raises(ForecastError):
        validate_closes(bad, 10)


def test_validate_rejects_non_numeric_and_2d() -> None:
    with pytest.raises(ForecastError):
        validate_closes(["a", "b"], 1)
    with pytest.raises(ForecastError):
        validate_closes([[1.0, 2.0], [3.0, 4.0]], 1)


def test_fit_rejects_too_little_data() -> None:
    with pytest.raises(ForecastError):
        fit_ridge(np.linspace(1, 2, 50))


def test_fit_rejects_bad_hyperparameters(closes: np.ndarray) -> None:
    with pytest.raises(ForecastError):
        fit_ridge(closes, n_lags=0)
    with pytest.raises(ForecastError):
        fit_ridge(closes, alpha=-1.0)


def test_constant_series_does_not_divide_by_zero() -> None:
    model = fit_ridge(np.full(300, 100.0))
    np.testing.assert_allclose(model.predict(np.full(40, 100.0)), 100.0)


def test_extreme_history_raises_forecast_error(closes: np.ndarray) -> None:
    model = fit_ridge(closes)
    spike = np.concatenate([np.full(20, 1e-300), np.full(21, 1e300)])
    with pytest.raises(ForecastError):
        model.predict(spike)


def test_model_rejects_inconsistent_arrays(closes: np.ndarray) -> None:
    model = fit_ridge(closes)
    with pytest.raises(ForecastError):
        RidgeForecaster(
            n_lags=model.n_lags,
            horizon=model.horizon,
            coef=model.coef[:, :3],
            intercept=model.intercept,
            feat_mean=model.feat_mean,
            feat_scale=model.feat_scale,
        )
    with pytest.raises(ForecastError):
        RidgeForecaster(
            n_lags=model.n_lags,
            horizon=model.horizon,
            coef=model.coef,
            intercept=model.intercept,
            feat_mean=model.feat_mean,
            feat_scale=np.zeros(model.n_lags),
        )


def test_evaluate_holdout_reports_baseline(closes: np.ndarray) -> None:
    model = fit_ridge(closes[:-40])
    report = evaluate_holdout(model, closes, 20)
    for key in ("mae", "rmse", "mape"):
        assert report[key] >= 0
        assert report["baseline_naive"][key] >= 0
    assert report["n_origins"] == 20
    assert report["horizon"] == HORIZON


def test_evaluate_holdout_rejects_zero_origins(closes: np.ndarray) -> None:
    with pytest.raises(ForecastError):
        evaluate_holdout(fit_ridge(closes), closes, 0)
