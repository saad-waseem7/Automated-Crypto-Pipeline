"""Daily BTC pipeline: fetch, preprocess, train and evaluate, notify the API.

Each step runs one of our scripts with a dedicated Python (a venv that holds only
numpy), not inside Airflow's own interpreter. Airflow pins hundreds of
dependencies, so installing our stack beside it invites version conflicts.
Steps exchange data through files, never through XCom, and every step is
idempotent so Airflow retries and reruns are safe.

Configuration comes from environment variables (see the defaults below).
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from airflow.exceptions import AirflowSkipException
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, task

log = logging.getLogger(__name__)

PYTHON = os.environ.get("BTC_PYTHON", "/opt/airflow/train-venv/bin/python")
PROJECT_DIR = os.environ.get("BTC_PROJECT_DIR", "/opt/airflow/project")
DATA_DIR = Path(os.environ.get("BTC_DATA_DIR", "/opt/airflow/data"))
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")

RAW_PATH = DATA_DIR / "raw" / "btc_usdt_1d.csv"
FEATURES_PATH = DATA_DIR / "processed" / "features.csv"
REPORT_PATH = DATA_DIR / "reports" / "last_train.json"

STEP_ENV = {
    "MODEL_DIR": MODEL_DIR,
    "BTC_RAW_PATH": str(RAW_PATH),
    "BTC_FEATURES_PATH": str(FEATURES_PATH),
    "BTC_REPORT_PATH": str(REPORT_PATH),
}


def _script(module: str) -> str:
    return f"{shlex.quote(PYTHON)} -m {module}"


def _step(task_id: str, module: str, minutes: int) -> BashOperator:
    return BashOperator(
        task_id=task_id,
        bash_command=_script(module),
        cwd=PROJECT_DIR,
        env=STEP_ENV,
        append_env=True,
        execution_timeout=timedelta(minutes=minutes),
    )


@task(task_id="reload_api")
def reload_api() -> str:
    """Ask the API to load the newly promoted model.

    Best effort by design. Every API worker also polls for new versions, so a
    failed notification delays the switch by seconds and never loses it.
    """
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if report["status"] != "promoted":
        raise AirflowSkipException(f"model not promoted: {report['reason']}")

    url = os.environ.get("API_RELOAD_URL", "http://api:5000/reload")
    token = os.environ.get("RELOAD_TOKEN", "")
    if not token:
        log.warning("RELOAD_TOKEN not set, relying on API polling for the new model")
        return "polling"

    request = urllib.request.Request(
        url, method="POST", headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            answer = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        log.warning("reload call failed (%s), relying on API polling", exc)
        return "polling"
    log.info("API now serving %s", answer.get("model_version"))
    return str(answer.get("model_version"))


with DAG(
    dag_id="btc_etl_pipeline",
    description="Daily BTC-USDT ETL, retraining and API reload",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=1),
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "retry_exponential_backoff": True,
        "max_retry_delay": timedelta(minutes=30),
    },
    tags=["btc", "mlops"],
) as dag:
    fetch = _step("fetch_data", "src.etl_fetch", minutes=20)
    preprocess = _step("preprocess_features", "src.features", minutes=10)
    train = _step("train_evaluate", "src.train_forecast", minutes=20)

    fetch >> preprocess >> train >> reload_api()
