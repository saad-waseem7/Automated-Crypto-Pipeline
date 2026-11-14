"""Flask REST API serving the 7-day BTC-USD close forecast.

Run in production with ``gunicorn -c app/gunicorn.conf.py app.app:app``.
"""

from __future__ import annotations

import hmac
import json
import logging
import time
from datetime import date, timedelta
from typing import Any

from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import HTTPException

from app.config import Settings
from app.model_store import ModelStore
from app.schemas import ValidationError, parse_predict_request
from src.forecast_core import ForecastError

log = logging.getLogger("btc_api")


class ApiError(Exception):
    """An error with a stable machine-readable code and a safe client message."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        details: list[str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details
        self.headers = headers or {}


def _error_response(
    status: int,
    code: str,
    message: str,
    details: list[str] | None = None,
    headers: dict[str, str] | None = None,
) -> Response:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    response = jsonify(body)
    response.status_code = status
    for key, value in (headers or {}).items():
        response.headers[key] = value
    return response


def _read_json_body() -> Any:
    """Decode the body. Empty body yields None; malformed JSON is a 400."""
    raw = request.get_data()
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise ApiError(400, "invalid_json", "request body is not valid JSON") from exc


def create_app(
    settings: Settings | None = None, store: ModelStore | None = None
) -> Flask:
    cfg = settings or Settings.from_env()
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    model_store = store or ModelStore(cfg.model_dir, cfg.poll_seconds)
    started = time.monotonic()

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_body_bytes
    app.json.sort_keys = False  # type: ignore[attr-defined]

    @app.before_request
    def _start_timer() -> None:
        request.environ["btc.start"] = time.monotonic()

    @app.after_request
    def _log_request(response: Response) -> Response:
        now = time.monotonic()
        elapsed_ms = (now - request.environ.get("btc.start", now)) * 1000
        log.info(
            "%s %s -> %d (%.1f ms)",
            request.method,
            request.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.get("/health")
    def health() -> Response:
        """Liveness: the process is up. Never fails because a model is missing,
        otherwise a cold start with an empty volume would crash-loop the pod."""
        bundle = model_store.get()
        return jsonify(
            {
                "status": "ok",
                "uptime_seconds": round(time.monotonic() - started, 1),
                "model_loaded": bundle is not None,
                "model_version": bundle.version if bundle else None,
                "model_stage": cfg.model_stage,
            }
        )

    @app.get("/ready")
    def ready() -> tuple[Response, int]:
        """Readiness: only receive traffic once a model is loaded."""
        bundle = model_store.get()
        if bundle is None:
            return jsonify({"status": "not_ready", "reason": "no model loaded"}), 503
        return jsonify({"status": "ready", "model_version": bundle.version}), 200

    @app.post("/predict")
    def predict() -> Response:
        req = parse_predict_request(_read_json_body(), cfg.max_history)
        bundle = model_store.get()
        if bundle is None:
            raise ApiError(503, "model_unavailable", "no model is loaded yet")

        if req.history is None:
            closes: Any = bundle.recent_closes
            base_date = bundle.last_date
            source = "model_window"
        else:
            closes = req.history
            assert req.history_end_date is not None  # guaranteed by validation
            base_date = req.history_end_date
            source = "request"
            if len(closes) < bundle.forecaster.min_history:
                raise ApiError(
                    400,
                    "invalid_history",
                    f"history needs at least {bundle.forecaster.min_history} values",
                )

        try:
            prices = bundle.forecaster.predict(closes)
        except ForecastError as exc:
            if source == "request":
                raise ApiError(400, "invalid_history", str(exc)) from exc
            raise

        horizon_dates: list[date] = [
            base_date + timedelta(days=i) for i in range(1, req.forecast_days + 1)
        ]
        return jsonify(
            {
                "model_version": bundle.version,
                "source": source,
                "base_date": base_date.isoformat(),
                "base_close": round(float(closes[-1]), 2),
                "forecast_days": req.forecast_days,
                "predictions": [
                    {"date": d.isoformat(), "predicted_close": round(float(p), 2)}
                    for d, p in zip(horizon_dates, prices[: req.forecast_days])
                ],
            }
        )

    @app.get("/metrics")
    def metrics() -> Response:
        bundle = model_store.get()
        if bundle is None:
            raise ApiError(503, "model_unavailable", "no model is loaded yet")
        return jsonify(
            {
                "model_version": bundle.version,
                "trained_at": bundle.trained_at,
                "metrics": bundle.metrics,
            }
        )

    @app.post("/reload")
    def reload_model() -> Response:
        """Optional accelerator: refresh this worker now. Other workers and pods
        converge on their own through polling, so callers must not rely on it."""
        if cfg.reload_token is None:
            raise ApiError(404, "not_found", "resource not found")
        header = request.headers.get("Authorization", "")
        supplied = header[7:] if header.startswith("Bearer ") else ""
        if not hmac.compare_digest(supplied.encode(), cfg.reload_token.encode()):
            raise ApiError(
                401,
                "unauthorized",
                "missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        changed = model_store.refresh(force=True)
        bundle = model_store.get()
        return jsonify(
            {
                "reloaded": changed,
                "model_version": bundle.version if bundle else None,
                "load_error": model_store.last_error is not None,
            }
        )

    @app.errorhandler(ApiError)
    def _handle_api_error(exc: ApiError) -> Response:
        return _error_response(
            exc.status, exc.code, exc.message, exc.details, exc.headers
        )

    @app.errorhandler(ValidationError)
    def _handle_validation(exc: ValidationError) -> Response:
        return _error_response(
            400, "validation_error", "request validation failed", exc.details
        )

    @app.errorhandler(HTTPException)
    def _handle_http(exc: HTTPException) -> Response:
        status = exc.code or 500
        code = (exc.name or "error").lower().replace(" ", "_")
        return _error_response(status, code, exc.description or exc.name)

    @app.errorhandler(Exception)
    def _handle_unexpected(exc: Exception) -> Response:
        log.exception("unhandled error on %s %s", request.method, request.path)
        return _error_response(500, "internal_error", "internal server error")

    return app


app = create_app()
