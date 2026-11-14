"""On-disk model artifact contract shared by trainers (writers) and the API (reader).

Layout under MODEL_DIR::

    CURRENT                      text file holding the active version name
    versions/<version>/model.npz        ridge coefficients (no pickle)
    versions/<version>/metadata.json    version info and the latest closes window
    versions/<version>/metrics.json     validation metrics and naive baseline

Writers build a version in a temp directory, rename it into place, and only
then replace CURRENT with an atomic rename. Readers therefore never observe a
half-written model, even when the volume is shared between pods.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.forecast_core import FloatArray, ForecastError, RidgeForecaster

SCHEMA_VERSION = 1
CURRENT_FILE = "CURRENT"
VERSIONS_DIR = "versions"
MODEL_FILE = "model.npz"
METADATA_FILE = "metadata.json"
METRICS_FILE = "metrics.json"
REQUIRED_METRICS = ("mae", "rmse", "mape")

_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing, malformed, or unsafe to load."""


@dataclass(frozen=True)
class ModelBundle:
    """Everything the API needs to answer a prediction request."""

    version: str
    trained_at: str
    last_date: date
    recent_closes: FloatArray
    forecaster: RidgeForecaster
    metrics: dict[str, Any]


def validate_version(version: str) -> str:
    """Reject names that could escape the versions directory."""
    if not _VERSION_RE.fullmatch(version):
        raise ArtifactError(f"invalid version name: {version!r}")
    return version


def new_version_name(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return stamp


def _fsync_write(path: Path, data: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _check_metrics(metrics: dict[str, Any]) -> None:
    for key in REQUIRED_METRICS:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ArtifactError(f"metrics.{key} must be a number")
        if not math.isfinite(value):
            raise ArtifactError(f"metrics.{key} must be finite")


def write_version(
    model_dir: Path,
    forecaster: RidgeForecaster,
    closes: FloatArray,
    last_date: date,
    metrics: dict[str, Any],
    *,
    version: str | None = None,
    promote: bool = True,
) -> str:
    """Persist a model version and optionally make it current. Returns the name."""
    name = validate_version(version or new_version_name())
    _check_metrics(metrics)
    if closes.size < forecaster.min_history:
        raise ArtifactError("closes window shorter than model min_history")

    versions = model_dir / VERSIONS_DIR
    versions.mkdir(parents=True, exist_ok=True)
    target = versions / name
    if target.exists():
        raise ArtifactError(f"version already exists: {name}")

    staging = Path(tempfile.mkdtemp(prefix=f".tmp-{name}-", dir=versions))
    try:
        np.savez(
            staging / MODEL_FILE,
            coef=forecaster.coef,
            intercept=forecaster.intercept,
            feat_mean=forecaster.feat_mean,
            feat_scale=forecaster.feat_scale,
        )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "version": name,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "last_date": last_date.isoformat(),
            "n_lags": forecaster.n_lags,
            "horizon": forecaster.horizon,
            "recent_closes": [
                float(v) for v in closes[-forecaster.min_history :].tolist()
            ],
        }
        _fsync_write(staging / METADATA_FILE, json.dumps(metadata))
        _fsync_write(staging / METRICS_FILE, json.dumps(metrics, allow_nan=False))
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if promote:
        promote_version(model_dir, name)
    return name


def promote_version(model_dir: Path, version: str) -> None:
    """Atomically point CURRENT at an existing version."""
    validate_version(version)
    if not (model_dir / VERSIONS_DIR / version).is_dir():
        raise ArtifactError(f"cannot promote unknown version: {version}")
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=".CURRENT-", dir=model_dir)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write(version)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, model_dir / CURRENT_FILE)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def read_current(model_dir: Path) -> str | None:
    """Return the active version name, or None if nothing has been promoted."""
    try:
        raw = (model_dir / CURRENT_FILE).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise ArtifactError(f"cannot read {CURRENT_FILE}: {exc}") from exc
    return validate_version(raw) if raw else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactError(f"{path.name} must contain a JSON object")
    return data


def load_bundle(model_dir: Path, version: str) -> ModelBundle:
    """Load and fully validate one version. Never executes pickled code."""
    validate_version(version)
    root = model_dir / VERSIONS_DIR / version
    if not root.is_dir():
        raise ArtifactError(f"version directory missing: {version}")

    metadata = _read_json(root / METADATA_FILE)
    metrics = _read_json(root / METRICS_FILE)
    try:
        if metadata["schema_version"] != SCHEMA_VERSION:
            raise ArtifactError(f"unsupported schema {metadata['schema_version']}")
        if metadata["version"] != version:
            raise ArtifactError("metadata version does not match directory name")
        last_date = date.fromisoformat(metadata["last_date"])
        closes = np.asarray(metadata["recent_closes"], dtype=np.float64)
        n_lags = int(metadata["n_lags"])
        horizon = int(metadata["horizon"])
        trained_at = str(metadata["trained_at"])
        with np.load(root / MODEL_FILE, allow_pickle=False) as arrays:
            forecaster = RidgeForecaster(
                n_lags=n_lags,
                horizon=horizon,
                coef=np.asarray(arrays["coef"], dtype=np.float64),
                intercept=np.asarray(arrays["intercept"], dtype=np.float64),
                feat_mean=np.asarray(arrays["feat_mean"], dtype=np.float64),
                feat_scale=np.asarray(arrays["feat_scale"], dtype=np.float64),
            )
        if closes.shape != (forecaster.min_history,):
            raise ArtifactError("recent_closes length does not match n_lags")
        if not np.isfinite(closes).all() or (closes <= 0).any():
            raise ArtifactError("recent_closes must be finite and positive")
    except ArtifactError:
        raise
    except (KeyError, TypeError, ValueError, OSError, ForecastError) as exc:
        raise ArtifactError(f"invalid artifact {version}: {exc}") from exc

    _check_metrics(metrics)
    return ModelBundle(
        version=version,
        trained_at=trained_at,
        last_date=last_date,
        recent_closes=closes,
        forecaster=forecaster,
        metrics=metrics,
    )


def prune_versions(model_dir: Path, keep: int) -> list[str]:
    """Delete all but the newest ``keep`` versions. Never deletes CURRENT."""
    if keep < 1:
        raise ArtifactError("keep must be at least 1")
    versions = model_dir / VERSIONS_DIR
    if not versions.is_dir():
        return []
    current = read_current(model_dir)
    names = sorted(
        p.name
        for p in versions.iterdir()
        if p.is_dir() and _VERSION_RE.fullmatch(p.name)
    )
    doomed = [n for n in names[:-keep] if n != current]
    for name in doomed:
        shutil.rmtree(versions / name, ignore_errors=True)
    return doomed
