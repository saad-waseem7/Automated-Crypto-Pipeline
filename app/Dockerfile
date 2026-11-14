# syntax=docker/dockerfile:1
#
# Build from the repository root so src/ is inside the build context:
#   docker build -f app/Dockerfile -t btc-forecast-api .
#
# Stages: builder -> base -> seed (dev/CI only) -> runtime (default, last).

ARG PYTHON_VERSION=3.11

# ---- builder: resolve and install dependencies into a relocatable venv ----
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
# Requirements first so this layer is reused until dependencies change.
COPY app/requirements.txt /tmp/requirements.txt
# pip and setuptools are only needed to install; dropping them saves about 25 MB.
RUN pip install -r /tmp/requirements.txt \
    && pip uninstall -y pip setuptools

# ---- base: shared runtime foundation, no pip, no compilers ----
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    MODEL_DIR=/app/models \
    PORT=5000
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home \
        --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY app/ ./app/
COPY src/ ./src/
# Created and owned here so a fresh named volume inherits the right owner.
RUN mkdir -p /app/models && chown app:app /app/models

# ---- seed: adds the synthetic model generator for local and CI smoke tests ----
FROM base AS seed
COPY scripts/ ./scripts/
USER 10001:10001
CMD ["python", "-m", "scripts.seed_artifacts", "--model-dir", "/app/models"]

# ---- runtime: what ships. Keep this stage last so it is the default target ----
FROM base AS runtime
USER 10001:10001
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '5000'), timeout=2)"]
# Exec form so SIGTERM reaches gunicorn and shutdown is graceful.
CMD ["gunicorn", "-c", "app/gunicorn.conf.py", "app.app:app"]
