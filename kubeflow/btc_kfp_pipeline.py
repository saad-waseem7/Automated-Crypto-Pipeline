"""Kubeflow Pipelines (KFP v2) definition of the BTC forecasting workflow.

Three container components run the same scripts as the Airflow DAG, from the same
image published by CI, so both orchestrators execute identical code. Data flows
between steps as KFP artifacts.

Compile to the committed YAML spec with::

    pip install -r requirements-kfp.txt
    python kubeflow/btc_kfp_pipeline.py

The spec is compiled and checked in CI but not deployed: a full Kubeflow
installation does not fit a 16 GB laptop next to Minikube and Airflow.
"""

# Do not add "from __future__ import annotations" here: KFP reads the annotations at
# runtime to build the component interface and cannot parse them as strings.
from pathlib import Path

from kfp import compiler, dsl

IMAGE = "ghcr.io/saad-waseem7/automated-crypto-pipeline:latest"
SPEC_PATH = Path(__file__).with_name("btc_kfp_pipeline.yaml")


@dsl.container_component
def ingest_data(raw_data: dsl.Output[dsl.Dataset]):
    """Fetch closed daily BTCUSDT candles into a raw CSV."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "src.etl_fetch"],
        args=["--raw-path", raw_data.path],
    )


@dsl.container_component
def build_features(
    raw_data: dsl.Input[dsl.Dataset],
    features: dsl.Output[dsl.Dataset],
):
    """Clean the series and compute MA7, MA30, volatility and training targets."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "src.features"],
        args=["--raw-path", raw_data.path, "--out-path", features.path],
    )


@dsl.container_component
def train_model(
    features: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model],
    report: dsl.Output[dsl.Artifact],
):
    """Train, evaluate against the naive baseline, and publish a model version."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "src.train_forecast"],
        args=[
            "--features-path",
            features.path,
            "--model-dir",
            model.path,
            "--report-path",
            report.path,
        ],
    )


@dsl.pipeline(
    name="btc-forecast-pipeline",
    description="Ingest BTC-USDT data, build features, train and evaluate the "
    "7-day close forecaster.",
)
def btc_pipeline() -> None:
    ingest = ingest_data()
    ingest.set_retry(num_retries=2, backoff_duration="60s", backoff_factor=2.0)

    features = build_features(raw_data=ingest.outputs["raw_data"])

    train = train_model(features=features.outputs["features"])
    train.set_cpu_limit("1")
    train.set_memory_limit("1G")


def compile_pipeline(path: Path = SPEC_PATH) -> Path:
    compiler.Compiler().compile(pipeline_func=btc_pipeline, package_path=str(path))
    return path


if __name__ == "__main__":
    print(f"compiled {compile_pipeline()}")
