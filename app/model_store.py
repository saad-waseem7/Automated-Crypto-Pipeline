"""Per-process model holder that converges to the newest promoted version.

Why polling and not push: behind a Service with several replicas, each running
several gunicorn workers, an HTTP "reload" call reaches exactly one worker.
Every worker instead re-checks the tiny CURRENT file at most every
``poll_seconds``, so all replicas converge without coordination. The reload
endpoint is only an accelerator on top of that.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from src.artifacts import ArtifactError, ModelBundle, load_bundle, read_current

log = logging.getLogger(__name__)


class ModelStore:
    def __init__(
        self,
        model_dir: Path,
        poll_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._dir = model_dir
        self._poll = poll_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._bundle: ModelBundle | None = None
        self._failed_version: str | None = None
        self._last_error: str | None = None
        self._last_check = float("-inf")

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def get(self) -> ModelBundle | None:
        """Return the active bundle, refreshing first if the poll interval elapsed."""
        if self._clock() - self._last_check >= self._poll:
            self.refresh()
        return self._bundle

    def refresh(self, *, force: bool = False) -> bool:
        """Load CURRENT if it changed. Returns True when a new bundle was installed.

        A failure never discards the model already being served, and a version
        that failed to load is not retried until CURRENT points elsewhere.
        """
        with self._lock:
            self._last_check = self._clock()
            try:
                current = read_current(self._dir)
            except ArtifactError as exc:
                self._record_failure(None, exc)
                return False
            if current is None:
                return False
            loaded = self._bundle.version if self._bundle else None
            if not force and (current == loaded or current == self._failed_version):
                return False
            try:
                bundle = load_bundle(self._dir, current)
            except ArtifactError as exc:
                self._record_failure(current, exc)
                return False
            self._bundle = bundle
            self._failed_version = None
            self._last_error = None
            log.info("loaded model version %s", current)
            return True

    def _record_failure(self, version: str | None, exc: ArtifactError) -> None:
        self._failed_version = version
        self._last_error = str(exc)
        log.error("model load failed (version=%s): %s", version, exc)
