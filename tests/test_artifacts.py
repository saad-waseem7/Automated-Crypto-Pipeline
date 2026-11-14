from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from src.artifacts import (
    ArtifactError,
    load_bundle,
    promote_version,
    prune_versions,
    read_current,
    validate_version,
    write_version,
)
from src.forecast_core import fit_ridge

Publish = Callable[..., str]


def test_roundtrip(tmp_path: Path, publish: Publish) -> None:
    publish(tmp_path, "v1")
    assert read_current(tmp_path) == "v1"
    bundle = load_bundle(tmp_path, "v1")
    assert bundle.version == "v1"
    assert bundle.last_date == date(2026, 1, 31)
    assert bundle.recent_closes.shape == (bundle.forecaster.min_history,)
    assert {"mae", "rmse", "mape"} <= set(bundle.metrics)


def test_read_current_empty(tmp_path: Path) -> None:
    assert read_current(tmp_path) is None
    (tmp_path / "CURRENT").write_text("   \n")
    assert read_current(tmp_path) is None


@pytest.mark.parametrize("name", ["../etc", "a/b", "", ".hidden", "x" * 80, "a b"])
def test_version_name_rejected(name: str) -> None:
    with pytest.raises(ArtifactError):
        validate_version(name)


def test_current_with_traversal_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "CURRENT").write_text("../../etc/passwd")
    with pytest.raises(ArtifactError):
        read_current(tmp_path)


def test_duplicate_version_refused(tmp_path: Path, publish: Publish) -> None:
    publish(tmp_path, "v1")
    with pytest.raises(ArtifactError):
        publish(tmp_path, "v1")


def test_unpromoted_version_not_current(tmp_path: Path, publish: Publish) -> None:
    publish(tmp_path, "v1")
    publish(tmp_path, "v2", promote=False)
    assert read_current(tmp_path) == "v1"
    promote_version(tmp_path, "v2")
    assert read_current(tmp_path) == "v2"


def test_promote_unknown_version(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(ArtifactError):
        promote_version(tmp_path, "nope")


def test_failed_write_leaves_no_staging_dir(
    tmp_path: Path, closes: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = fit_ridge(closes)
    metrics = {"mae": 1.0, "rmse": 1.0, "mape": 1.0}

    def failing_replace(*_: object, **__: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("src.artifacts.os.replace", failing_replace)
    with pytest.raises(OSError):
        write_version(tmp_path, model, closes, date(2026, 1, 1), metrics)
    assert list((tmp_path / "versions").iterdir()) == []
    assert read_current(tmp_path) is None


def test_nan_metric_rejected(tmp_path: Path, closes: np.ndarray) -> None:
    model = fit_ridge(closes)
    metrics = {"mae": float("nan"), "rmse": 1.0, "mape": 1.0}
    with pytest.raises(ArtifactError):
        write_version(tmp_path, model, closes, date(2026, 1, 1), metrics)


def test_short_closes_rejected(tmp_path: Path, closes: np.ndarray) -> None:
    model = fit_ridge(closes)
    metrics = {"mae": 1.0, "rmse": 1.0, "mape": 1.0}
    with pytest.raises(ArtifactError):
        write_version(tmp_path, model, closes[:5], date(2026, 1, 1), metrics)


def test_missing_version_dir(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        load_bundle(tmp_path, "ghost")


@pytest.mark.parametrize("victim", ["model.npz", "metadata.json", "metrics.json"])
def test_corrupt_files_raise_artifact_error(
    tmp_path: Path, publish: Publish, victim: str
) -> None:
    publish(tmp_path, "v1")
    (tmp_path / "versions" / "v1" / victim).write_bytes(b"garbage")
    with pytest.raises(ArtifactError):
        load_bundle(tmp_path, "v1")


def test_missing_model_file(tmp_path: Path, publish: Publish) -> None:
    publish(tmp_path, "v1")
    (tmp_path / "versions" / "v1" / "model.npz").unlink()
    with pytest.raises(ArtifactError):
        load_bundle(tmp_path, "v1")


def test_metadata_edits_are_validated(tmp_path: Path, publish: Publish) -> None:
    publish(tmp_path, "v1")
    path = tmp_path / "versions" / "v1" / "metadata.json"
    good = json.loads(path.read_text())

    for field, value in [
        ("schema_version", 99),
        ("version", "other"),
        ("last_date", "not-a-date"),
        ("recent_closes", [1.0, 2.0]),
        ("recent_closes", [-1.0] * 31),
    ]:
        path.write_text(json.dumps({**good, field: value}))
        with pytest.raises(ArtifactError):
            load_bundle(tmp_path, "v1")

    path.write_text(json.dumps(["not", "an", "object"]))
    with pytest.raises(ArtifactError):
        load_bundle(tmp_path, "v1")


def test_pickled_npz_is_not_executed(tmp_path: Path, publish: Publish) -> None:
    publish(tmp_path, "v1")
    target = tmp_path / "versions" / "v1" / "model.npz"
    np.savez(
        target,
        coef=np.array([object()], dtype=object),
        intercept=np.zeros(1),
        feat_mean=np.zeros(1),
        feat_scale=np.ones(1),
    )
    with pytest.raises(ArtifactError):
        load_bundle(tmp_path, "v1")


def test_prune_keeps_newest_and_current(tmp_path: Path, publish: Publish) -> None:
    for name in ["v1", "v2", "v3", "v4"]:
        publish(tmp_path, name)
    promote_version(tmp_path, "v1")
    removed = prune_versions(tmp_path, keep=2)
    assert removed == ["v2"]
    remaining = sorted(p.name for p in (tmp_path / "versions").iterdir())
    assert remaining == ["v1", "v3", "v4"]


def test_prune_edge_cases(tmp_path: Path) -> None:
    assert prune_versions(tmp_path, keep=1) == []
    with pytest.raises(ArtifactError):
        prune_versions(tmp_path, keep=0)
