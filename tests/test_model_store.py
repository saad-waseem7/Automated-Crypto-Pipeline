from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from app.model_store import ModelStore
from src.artifacts import promote_version

Publish = Callable[..., str]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_no_model_returns_none(tmp_path: Path) -> None:
    store = ModelStore(tmp_path, poll_seconds=0)
    assert store.get() is None
    assert store.last_error is None


def test_loads_current(tmp_path: Path, publish: Publish) -> None:
    publish(tmp_path, "v1")
    bundle = ModelStore(tmp_path, poll_seconds=0).get()
    assert bundle is not None and bundle.version == "v1"


def test_picks_up_new_version_only_after_poll_interval(
    tmp_path: Path, publish: Publish
) -> None:
    clock = FakeClock()
    publish(tmp_path, "v1")
    store = ModelStore(tmp_path, poll_seconds=10, clock=clock)
    first = store.get()
    assert first is not None and first.version == "v1"

    publish(tmp_path, "v2")
    clock.now += 5
    stale = store.get()
    assert stale is not None and stale.version == "v1"

    clock.now += 6
    fresh = store.get()
    assert fresh is not None and fresh.version == "v2"


def test_broken_version_keeps_serving_old_and_is_not_retried(
    tmp_path: Path, publish: Publish
) -> None:
    clock = FakeClock()
    publish(tmp_path, "v1")
    store = ModelStore(tmp_path, poll_seconds=1, clock=clock)
    assert store.get() is not None

    publish(tmp_path, "v2")
    (tmp_path / "versions" / "v2" / "model.npz").write_bytes(b"junk")
    clock.now += 2
    served = store.get()
    assert served is not None and served.version == "v1"
    assert store.last_error is not None

    # Repair the files: the store must not retry v2 until CURRENT changes
    # or a forced refresh is requested.
    publish(tmp_path, "v3", promote=False)
    clock.now += 2
    store.get()
    assert store.refresh() is False

    promote_version(tmp_path, "v3")
    clock.now += 2
    recovered = store.get()
    assert recovered is not None and recovered.version == "v3"
    assert store.last_error is None


def test_force_reloads_same_version(tmp_path: Path, publish: Publish) -> None:
    publish(tmp_path, "v1")
    store = ModelStore(tmp_path, poll_seconds=100)
    store.get()
    assert store.refresh() is False
    assert store.refresh(force=True) is True


def test_unreadable_current_is_recorded(tmp_path: Path) -> None:
    (tmp_path / "CURRENT").write_text("../bad")
    store = ModelStore(tmp_path, poll_seconds=0)
    assert store.get() is None
    assert store.last_error is not None
