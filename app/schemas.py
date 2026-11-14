"""Strict validation of the /predict request body."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from src.forecast_core import HORIZON

ALLOWED_FIELDS = frozenset({"forecast_days", "history", "history_end_date"})


class ValidationError(ValueError):
    """Carries every problem found so clients can fix them in one round trip."""

    def __init__(self, details: list[str]) -> None:
        super().__init__("; ".join(details))
        self.details = details


@dataclass(frozen=True)
class PredictRequest:
    forecast_days: int
    history: list[float] | None
    history_end_date: date | None


def _is_number(value: Any) -> bool:
    # bool is a subclass of int in Python; JSON true/false must not pass as numbers.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_predict_request(payload: Any, max_history: int) -> PredictRequest:
    """Validate a decoded JSON body. ``None`` means an empty body (all defaults)."""
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValidationError(["body must be a JSON object"])

    problems: list[str] = []
    unknown = sorted(set(payload) - ALLOWED_FIELDS)
    if unknown:
        problems.append(f"unknown fields: {', '.join(unknown)}")

    days = payload.get("forecast_days", HORIZON)
    if isinstance(days, bool) or not isinstance(days, int):
        problems.append("forecast_days must be an integer")
        days = HORIZON
    elif not 1 <= days <= HORIZON:
        problems.append(f"forecast_days must be between 1 and {HORIZON}")

    history = _parse_history(payload.get("history"), max_history, problems)
    end_date = _parse_end_date(payload.get("history_end_date"), history, problems)

    if problems:
        raise ValidationError(problems)
    return PredictRequest(
        forecast_days=days, history=history, history_end_date=end_date
    )


def _parse_history(
    raw: Any, max_history: int, problems: list[str]
) -> list[float] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        problems.append("history must be an array of numbers")
        return None
    if len(raw) > max_history:
        problems.append(f"history may contain at most {max_history} values")
        return None
    values: list[float] = []
    for index, item in enumerate(raw):
        if not _is_number(item):
            problems.append(f"history[{index}] must be a number")
            return None
        try:
            number = float(item)
        except OverflowError:
            problems.append(f"history[{index}] is out of range")
            return None
        if not math.isfinite(number) or number <= 0:
            problems.append(f"history[{index}] must be a finite positive number")
            return None
        values.append(number)
    return values


def _parse_end_date(
    raw: Any, history: list[float] | None, problems: list[str]
) -> date | None:
    if raw is None:
        if history is not None:
            problems.append("history_end_date is required when history is given")
        return None
    if history is None:
        problems.append("history_end_date is only valid together with history")
        return None
    if not isinstance(raw, str):
        problems.append("history_end_date must be a YYYY-MM-DD string")
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        problems.append("history_end_date must be a valid YYYY-MM-DD date")
        return None
