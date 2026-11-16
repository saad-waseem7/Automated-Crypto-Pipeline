from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from flask import Flask
from flask.testing import FlaskClient

from app.app import create_app
from src.forecast_core import ForecastError
from tests.conftest import LAST_DATE, TOKEN, make_settings

Publish = Callable[..., str]


def err_code(response: Any) -> str:
    return str(response.get_json()["error"]["code"])


class TestHealthAndReady:
    def test_health_with_model(self, client: FlaskClient) -> None:
        body = client.get("/health").get_json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_version"] == "v1"
        assert body["uptime_seconds"] >= 0

    def test_json_responses_carry_security_headers(self, client: FlaskClient) -> None:
        response = client.get("/health")
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["Cache-Control"] == "no-store"
        assert client.get("/nope").headers["X-Content-Type-Options"] == "nosniff"

    def test_health_without_model_is_still_200(self, empty_client: FlaskClient) -> None:
        response = empty_client.get("/health")
        assert response.status_code == 200
        assert response.get_json()["model_loaded"] is False

    def test_ready_flips_with_model(
        self, empty_client: FlaskClient, client: FlaskClient
    ) -> None:
        assert empty_client.get("/ready").status_code == 503
        assert client.get("/ready").status_code == 200


class TestPredict:
    def test_default_is_seven_days_from_stored_window(
        self, client: FlaskClient
    ) -> None:
        response = client.post("/predict")
        assert response.status_code == 200
        body = response.get_json()
        assert body["source"] == "model_window"
        assert body["forecast_days"] == 7
        assert body["base_date"] == LAST_DATE.isoformat()
        preds = body["predictions"]
        assert len(preds) == 7
        assert preds[0]["date"] == (LAST_DATE + timedelta(days=1)).isoformat()
        assert preds[-1]["date"] == (LAST_DATE + timedelta(days=7)).isoformat()
        assert all(p["predicted_close"] > 0 for p in preds)

    @pytest.mark.parametrize("days", [1, 3, 7])
    def test_forecast_days(self, client: FlaskClient, days: int) -> None:
        body = client.post("/predict", json={"forecast_days": days}).get_json()
        assert len(body["predictions"]) == days

    def test_empty_json_object_uses_defaults(self, client: FlaskClient) -> None:
        assert len(client.post("/predict", json={}).get_json()["predictions"]) == 7

    @pytest.mark.parametrize("days", [0, 8, -1, "7", 7.5, True, None, [7]])
    def test_bad_forecast_days(self, client: FlaskClient, days: Any) -> None:
        response = client.post("/predict", json={"forecast_days": days})
        assert response.status_code == 400
        assert err_code(response) == "validation_error"

    def test_unknown_field_rejected(self, client: FlaskClient) -> None:
        response = client.post("/predict", json={"days": 3})
        assert response.status_code == 400
        assert "unknown fields" in response.get_json()["error"]["details"][0]

    def test_multiple_problems_reported_together(self, client: FlaskClient) -> None:
        response = client.post("/predict", json={"forecast_days": 99, "extra": 1})
        assert len(response.get_json()["error"]["details"]) == 2

    @pytest.mark.parametrize("raw", [b"{not json", b"[1,2,3]", b'"text"', b"42"])
    def test_malformed_or_non_object_body(
        self, client: FlaskClient, raw: bytes
    ) -> None:
        response = client.post("/predict", data=raw, content_type="application/json")
        assert response.status_code == 400

    def test_deeply_nested_json_is_400_not_500(self, client: FlaskClient) -> None:
        response = client.post(
            "/predict", data=b"[" * 100_000, content_type="application/json"
        )
        assert response.status_code == 400

    def test_custom_history(self, client: FlaskClient, history: list[float]) -> None:
        response = client.post(
            "/predict",
            json={
                "history": history,
                "history_end_date": "2026-06-01",
                "forecast_days": 2,
            },
        )
        body = response.get_json()
        assert response.status_code == 200
        assert body["source"] == "request"
        assert body["base_date"] == "2026-06-01"
        assert body["predictions"][0]["date"] == "2026-06-02"
        assert body["base_close"] == round(history[-1], 2)

    def test_history_requires_end_date(
        self, client: FlaskClient, history: list[float]
    ) -> None:
        response = client.post("/predict", json={"history": history})
        assert response.status_code == 400

    def test_end_date_without_history(self, client: FlaskClient) -> None:
        response = client.post("/predict", json={"history_end_date": "2026-06-01"})
        assert response.status_code == 400

    @pytest.mark.parametrize("bad_date", ["2026-13-01", "tomorrow", 20260601, ""])
    def test_bad_end_date(
        self, client: FlaskClient, history: list[float], bad_date: Any
    ) -> None:
        response = client.post(
            "/predict", json={"history": history, "history_end_date": bad_date}
        )
        assert response.status_code == 400

    def test_history_too_short(self, client: FlaskClient, history: list[float]) -> None:
        response = client.post(
            "/predict",
            json={"history": history[:5], "history_end_date": "2026-06-01"},
        )
        assert response.status_code == 400
        assert err_code(response) == "invalid_history"

    @pytest.mark.parametrize(
        "poison",
        ["abc", None, True, -5, 0, [1], {"a": 1}],
    )
    def test_history_bad_element(
        self, client: FlaskClient, history: list[float], poison: Any
    ) -> None:
        bad = history[:-1] + [poison]
        response = client.post(
            "/predict", json={"history": bad, "history_end_date": "2026-06-01"}
        )
        assert response.status_code == 400

    @pytest.mark.parametrize("literal", [b"NaN", b"Infinity", b"-Infinity", b"1e999"])
    def test_non_finite_numbers_rejected(
        self, client: FlaskClient, history: list[float], literal: bytes
    ) -> None:
        values = b",".join(repr(v).encode() for v in history[:-1]) + b"," + literal
        raw = b'{"history_end_date":"2026-06-01","history":[' + values + b"]}"
        response = client.post("/predict", data=raw, content_type="application/json")
        assert response.status_code == 400

    def test_huge_integer_is_400_not_500(
        self, client: FlaskClient, history: list[float]
    ) -> None:
        values = b",".join(repr(v).encode() for v in history[:-1])
        raw = (
            b'{"history_end_date":"2026-06-01","history":['
            + values
            + b",1"
            + b"0" * 400
            + b"]}"
        )
        response = client.post("/predict", data=raw, content_type="application/json")
        assert response.status_code == 400

    def test_history_not_a_list(self, client: FlaskClient) -> None:
        response = client.post(
            "/predict", json={"history": "1,2,3", "history_end_date": "2026-06-01"}
        )
        assert response.status_code == 400

    def test_history_over_limit(self, client: FlaskClient) -> None:
        response = client.post(
            "/predict",
            json={"history": [1.0] * 1001, "history_end_date": "2026-06-01"},
        )
        assert response.status_code == 400

    def test_history_that_overflows_forecast_is_400(self, client: FlaskClient) -> None:
        spike = [1e-300] * 20 + [1e300] * 21
        response = client.post(
            "/predict", json={"history": spike, "history_end_date": "2026-06-01"}
        )
        assert response.status_code == 400
        assert err_code(response) == "invalid_history"

    def test_no_model_is_503(self, empty_client: FlaskClient) -> None:
        response = empty_client.post("/predict")
        assert response.status_code == 503
        assert err_code(response) == "model_unavailable"

    def test_oversized_body_is_413(self, tmp_path: Path, publish: Publish) -> None:
        publish(tmp_path)
        small = create_app(make_settings(tmp_path, max_body_bytes=1024))
        response = small.test_client().post(
            "/predict", data=b" " * 4096, content_type="application/json"
        )
        assert response.status_code == 413
        assert response.is_json

    def test_get_not_allowed(self, client: FlaskClient) -> None:
        response = client.get("/predict")
        assert response.status_code == 405
        assert response.is_json


class TestMetrics:
    def test_metrics(self, client: FlaskClient) -> None:
        body = client.get("/metrics").get_json()
        assert body["model_version"] == "v1"
        assert body["metrics"]["mae"] >= 0
        assert "baseline_naive" in body["metrics"]

    def test_metrics_without_model(self, empty_client: FlaskClient) -> None:
        assert empty_client.get("/metrics").status_code == 503


class TestReload:
    def test_picks_up_new_version(
        self, tmp_path: Path, publish: Publish, app_with_model: Flask
    ) -> None:
        client = app_with_model.test_client()
        publish(tmp_path, "v2")
        response = client.post("/reload", headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == 200
        assert response.get_json()["model_version"] == "v2"

    @pytest.mark.parametrize(
        "header", [None, "", "Bearer ", "Bearer wrong", f"Basic {TOKEN}", TOKEN]
    )
    def test_rejects_bad_credentials(
        self, client: FlaskClient, header: str | None
    ) -> None:
        headers = {} if header is None else {"Authorization": header}
        response = client.post("/reload", headers=headers)
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_disabled_without_token(self, tmp_path: Path) -> None:
        app = create_app(make_settings(tmp_path, reload_token=None))
        response = app.test_client().post("/reload")
        assert response.status_code == 404


class TestErrorHandling:
    def test_unknown_route_is_json_404(self, client: FlaskClient) -> None:
        response = client.get("/nope")
        assert response.status_code == 404
        assert response.is_json

    def test_unexpected_error_is_500_without_leak(
        self, app_with_model: Flask, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_: Any, **__: Any) -> None:
            raise RuntimeError("secret internal detail")

        monkeypatch.setattr("app.app.parse_predict_request", boom)
        response = app_with_model.test_client().post("/predict")
        assert response.status_code == 500
        assert "secret" not in response.get_data(as_text=True)
        assert err_code(response) == "internal_error"

    def test_forecast_error_on_stored_window_is_500(
        self, app_with_model: Flask, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_: Any, **__: Any) -> None:
            raise ForecastError("corrupt")

        monkeypatch.setattr("src.forecast_core.RidgeForecaster.predict", boom)
        response = app_with_model.test_client().post("/predict")
        assert response.status_code == 500

    def test_date_math_crosses_year_boundary(
        self, tmp_path: Path, publish: Publish
    ) -> None:
        publish(tmp_path)
        client = create_app(make_settings(tmp_path)).test_client()
        history = [100.0 + i * 0.1 for i in range(40)]
        body = client.post(
            "/predict",
            json={"history": history, "history_end_date": "2026-12-30"},
        ).get_json()
        dates = [p["date"] for p in body["predictions"]]
        assert dates[0] == date(2026, 12, 31).isoformat()
        assert dates[1] == "2027-01-01"
