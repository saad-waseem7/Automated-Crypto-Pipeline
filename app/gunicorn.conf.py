"""Gunicorn settings tuned for a small container (500m CPU, 512Mi limit).

Workers default to 2, not the textbook (2 x cores + 1): the container sees the
host's core count, so that formula would spawn far more workers than the CPU
limit can serve and risks an OOM kill. Override with WEB_CONCURRENCY.
"""

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))
worker_class = "gthread"
threads = int(os.environ.get("GUNICORN_THREADS", "2"))
timeout = 30
graceful_timeout = 30
keepalive = 5
# Default /tmp can be a disk-backed overlay in containers and stall heartbeats.
worker_tmp_dir = "/dev/shm"
accesslog = None  # the app logs each request itself
errorlog = "-"
