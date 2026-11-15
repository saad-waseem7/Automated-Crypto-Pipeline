"""Load test for the forecast API.

Headless example (drives the HPA past its 60% CPU target):
    locust -f loadtest/locustfile.py --headless -u 200 -r 20 -t 3m \
        --host http://$(minikube ip):30080
"""

from __future__ import annotations

from locust import HttpUser, between, task


class ForecastUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(8)
    def predict(self) -> None:
        with self.client.post("/predict", name="/predict", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"status {r.status_code}")
            elif len(r.json().get("predictions", [])) != 7:
                r.failure("expected 7 predictions")

    @task(1)
    def metrics(self) -> None:
        self.client.get("/metrics", name="/metrics")

    @task(1)
    def ready(self) -> None:
        self.client.get("/ready", name="/ready")
