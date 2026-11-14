from __future__ import annotations

from pathlib import Path

import pytest

from app.config import ConfigError, Settings


def test_defaults() -> None:
    cfg = Settings.from_env({})
    assert cfg.port == 5000
    assert cfg.log_level == "INFO"
    assert cfg.model_dir == Path("models")
    assert cfg.reload_token is None


def test_overrides() -> None:
    cfg = Settings.from_env(
        {
            "MODEL_DIR": "/app/models",
            "LOG_LEVEL": "debug",
            "PORT": "8080",
            "MODEL_STAGE": "staging",
            "RELOAD_TOKEN": "x" * 16,
            "MODEL_POLL_SECONDS": "0",
        }
    )
    assert cfg.model_dir == Path("/app/models")
    assert cfg.log_level == "DEBUG"
    assert cfg.port == 8080
    assert cfg.model_stage == "staging"
    assert cfg.poll_seconds == 0


@pytest.mark.parametrize(
    "env",
    [
        {"PORT": "abc"},
        {"PORT": "0"},
        {"PORT": "70000"},
        {"LOG_LEVEL": "LOUD"},
        {"RELOAD_TOKEN": "short"},
        {"MODEL_POLL_SECONDS": "-1"},
        {"MAX_HISTORY": "1"},
    ],
)
def test_invalid_values_fail_fast(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_empty_values_fall_back_to_defaults() -> None:
    cfg = Settings.from_env({"PORT": "", "RELOAD_TOKEN": ""})
    assert cfg.port == 5000
    assert cfg.reload_token is None
