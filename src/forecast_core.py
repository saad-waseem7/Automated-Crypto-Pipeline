"""Numpy-only direct multi-horizon forecaster for daily closing prices.

Training and serving import this same module, so feature construction cannot
drift between the two (no train/serve skew). Inference needs only numpy, which
keeps the API image small and avoids unpickling model files from a volume.

Model: for each horizon h in 1..H, a ridge regression maps the last
``n_lags`` daily log-returns to the cumulative log-return over h days.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

HORIZON = 7
N_LAGS = 30
MIN_TRAIN_SAMPLES = 100
_SCALE_FLOOR = 1e-12


class ForecastError(ValueError):
    """Raised when input data cannot be used to fit or forecast."""


def validate_closes(values: ArrayLike, min_length: int) -> FloatArray:
    """Return ``values`` as a 1-D float array after strict validation."""
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ForecastError("closes must be a sequence of numbers") from exc
    if arr.ndim != 1:
        raise ForecastError("closes must be one-dimensional")
    if arr.size < min_length:
        raise ForecastError(f"need at least {min_length} closes, got {arr.size}")
    if not np.isfinite(arr).all():
        raise ForecastError("closes must be finite numbers")
    if (arr <= 0).any():
        raise ForecastError("closes must be strictly positive")
    return arr


@dataclass(frozen=True)
class RidgeForecaster:
    """Fitted model. Immutable so it can be shared across threads safely."""

    n_lags: int
    horizon: int
    coef: FloatArray  # (n_lags, horizon)
    intercept: FloatArray  # (horizon,)
    feat_mean: FloatArray  # (n_lags,)
    feat_scale: FloatArray  # (n_lags,)

    def __post_init__(self) -> None:
        expected = {
            "coef": (self.n_lags, self.horizon),
            "intercept": (self.horizon,),
            "feat_mean": (self.n_lags,),
            "feat_scale": (self.n_lags,),
        }
        for name, shape in expected.items():
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ForecastError(f"{name} has shape {arr.shape}, expected {shape}")
            if not np.isfinite(arr).all():
                raise ForecastError(f"{name} contains non-finite values")
        if (self.feat_scale <= 0).any():
            raise ForecastError("feat_scale must be strictly positive")

    @property
    def min_history(self) -> int:
        """Closes needed to build one feature row (n_lags returns need n_lags+1)."""
        return self.n_lags + 1

    def predict(self, closes: ArrayLike) -> FloatArray:
        """Forecast the next ``horizon`` closes from the tail of ``closes``."""
        arr = validate_closes(closes, self.min_history)
        with np.errstate(all="ignore"):
            returns = np.diff(np.log(arr[-self.min_history :]))
            features = (returns - self.feat_mean) / self.feat_scale
            log_change = features @ self.coef + self.intercept
            prices: FloatArray = arr[-1] * np.exp(log_change)
        if not np.isfinite(prices).all():
            raise ForecastError("history produces a non-finite forecast")
        return prices


def fit_ridge(
    closes: ArrayLike,
    n_lags: int = N_LAGS,
    horizon: int = HORIZON,
    alpha: float = 100.0,
) -> RidgeForecaster:
    """Fit the forecaster on a closing-price history (oldest first)."""
    if n_lags < 1 or horizon < 1:
        raise ForecastError("n_lags and horizon must be positive")
    if alpha < 0:
        raise ForecastError("alpha must be non-negative")
    arr = validate_closes(closes, n_lags + horizon + MIN_TRAIN_SAMPLES)

    log_p = np.log(arr)
    returns = np.diff(log_p)
    n_samples = log_p.size - horizon - n_lags

    # Row k: returns ending at log_p[t], t = k + n_lags. Target: log_p[t+h] - log_p[t].
    x = sliding_window_view(returns, n_lags)[:n_samples]
    future = sliding_window_view(log_p, horizon + 1)[n_lags : n_lags + n_samples]
    y = future[:, 1:] - future[:, [0]]

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < _SCALE_FLOOR, 1.0, scale)
    xs = (x - mean) / scale
    y_mean = y.mean(axis=0)

    gram = xs.T @ xs + alpha * np.eye(n_lags)
    coef = np.linalg.solve(gram, xs.T @ (y - y_mean))
    return RidgeForecaster(
        n_lags=n_lags,
        horizon=horizon,
        coef=coef,
        intercept=y_mean,
        feat_mean=mean,
        feat_scale=scale,
    )


def _errors(actual: FloatArray, predicted: FloatArray) -> dict[str, float]:
    diff = predicted - actual
    return {
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt((diff**2).mean())),
        "mape": float((np.abs(diff) / actual).mean() * 100.0),
    }


def evaluate_holdout(
    model: RidgeForecaster, closes: ArrayLike, n_origins: int
) -> dict[str, Any]:
    """Score ``model`` on the last ``n_origins`` forecast origins.

    For every origin the model sees only closes up to that day and is compared
    with the next ``horizon`` actual closes. A naive "tomorrow equals today"
    baseline is scored on the same origins, because a model that cannot beat it
    is worthless however small its MAPE looks.
    """
    arr = validate_closes(closes, model.min_history + model.horizon + n_origins)
    if n_origins < 1:
        raise ForecastError("n_origins must be positive")
    first_origin = arr.size - model.horizon - n_origins
    actual_rows = []
    model_rows = []
    naive_rows = []
    for t in range(first_origin, first_origin + n_origins):
        history = arr[: t + 1]
        actual_rows.append(arr[t + 1 : t + 1 + model.horizon])
        model_rows.append(model.predict(history))
        naive_rows.append(np.full(model.horizon, history[-1]))
    actual = np.vstack(actual_rows)
    return {
        **_errors(actual, np.vstack(model_rows)),
        "n_origins": n_origins,
        "horizon": model.horizon,
        "baseline_naive": _errors(actual, np.vstack(naive_rows)),
    }
