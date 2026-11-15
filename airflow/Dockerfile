# syntax=docker/dockerfile:1
#
# Airflow for the local stack. Build from the repository root:
#   docker build -f airflow/Dockerfile -t btc-forecast-airflow .
#
# Our code runs in its own venv instead of Airflow's interpreter. Airflow pins
# hundreds of dependencies and installing ours beside them invites conflicts.
# The venv uses the same requirements file as the API image, so training and
# serving run identical library versions.
FROM apache/airflow:3.1.8-python3.11

COPY --chown=airflow:0 app/requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/airflow/train-venv \
    && /opt/airflow/train-venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt
