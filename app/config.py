"""Environment-driven settings. Fails fast at startup on malformed values."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConfigError(ValueError):
    """Raised when an environment variable has an invalid value."""


def _int(env: Mapping[str, str], key: str, default: int, low: int, high: int) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc
    if not low <= value <= high:
        raise ConfigError(f"{key} must be between {low} and {high}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path("models")
    model_stage: str = "production"
    log_level: str = "INFO"
    port: int = 5000
    poll_seconds: int = 10
    max_history: int = 5000
    max_body_bytes: int = 1_048_576
    reload_token: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source = os.environ if env is None else env
        level = source.get("LOG_LEVEL", "INFO").upper()
        if level not in _LOG_LEVELS:
            raise ConfigError(f"LOG_LEVEL must be one of {sorted(_LOG_LEVELS)}")
        token = source.get("RELOAD_TOKEN") or None
        if token is not None and len(token) < 16:
            raise ConfigError("RELOAD_TOKEN must be at least 16 characters")
        return cls(
            model_dir=Path(source.get("MODEL_DIR", "models")),
            model_stage=source.get("MODEL_STAGE", "production"),
            log_level=level,
            port=_int(source, "PORT", 5000, 1, 65535),
            poll_seconds=_int(source, "MODEL_POLL_SECONDS", 10, 0, 3600),
            max_history=_int(source, "MAX_HISTORY", 5000, 32, 100_000),
            max_body_bytes=_int(source, "MAX_BODY_BYTES", 1_048_576, 1024, 16_777_216),
            reload_token=token,
        )
