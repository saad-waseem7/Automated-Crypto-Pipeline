"""Static checks on k8s/: schema validity and cross-resource consistency.

These catch the mistakes that otherwise surface only as a pod stuck Pending:
a Service selector that matches nothing, a probe on the wrong path, a missing
PVC, a mutable image tag.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import kubernetes_validate
import pytest
import yaml

K8S = Path(__file__).resolve().parent.parent / "k8s"
VERSION = "1.31"
FILES = sorted(p for p in K8S.glob("*.yaml") if p.name != "kustomization.yaml")


def load_all() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in FILES:
        docs.extend(d for d in yaml.safe_load_all(path.read_text()) if d)
    return docs


def one(kind: str, name: str | None = None) -> dict[str, Any]:
    found = [
        d
        for d in load_all()
        if d["kind"] == kind and (name is None or d["metadata"]["name"] == name)
    ]
    assert len(found) == 1, f"expected exactly one {kind} {name}, got {len(found)}"
    return found[0]


@pytest.mark.parametrize(
    "doc", load_all(), ids=lambda d: f"{d['kind']}/{d['metadata']['name']}"
)
def test_schema_valid(doc: dict[str, Any]) -> None:
    kubernetes_validate.validate(doc, VERSION, strict=True)


def test_every_namespaced_resource_is_in_the_app_namespace() -> None:
    for doc in load_all():
        if doc["kind"] != "Namespace":
            assert doc["metadata"]["namespace"] == "btc-forecast", doc["metadata"]


def test_services_select_deployment_pods() -> None:
    deployment = one("Deployment")
    pod_labels = deployment["spec"]["template"]["metadata"]["labels"]
    assert deployment["spec"]["selector"]["matchLabels"].items() <= pod_labels.items()
    for svc in (d for d in load_all() if d["kind"] == "Service"):
        assert svc["spec"]["selector"].items() <= pod_labels.items()


def test_ingress_and_hpa_point_at_existing_objects() -> None:
    backend = one("Ingress")["spec"]["rules"][0]["http"]["paths"][0]["backend"]
    assert (
        backend["service"]["name"] == one("Service", "btc-forecast")["metadata"]["name"]
    )
    assert one("Ingress")["spec"]["rules"][0]["host"] == "btc-forecast.local"
    hpa = one("HorizontalPodAutoscaler")["spec"]
    assert hpa["scaleTargetRef"]["name"] == one("Deployment")["metadata"]["name"]
    assert hpa["minReplicas"] >= 2 and hpa["maxReplicas"] > hpa["minReplicas"]
    assert hpa["metrics"][0]["resource"]["target"]["averageUtilization"] == 60


def test_deployment_resources_and_probes() -> None:
    deployment = one("Deployment")
    assert deployment["spec"]["replicas"] >= 3
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    live = container["livenessProbe"]
    assert live["httpGet"]["path"] == "/health"
    assert live["initialDelaySeconds"] == 15 and live["periodSeconds"] == 10
    # Readiness deliberately uses /ready: 503 until a model is loaded.
    assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert container["resources"] == {
        "requests": {"cpu": "100m", "memory": "128Mi"},
        "limits": {"cpu": "500m", "memory": "512Mi"},
    }
    assert container["volumeMounts"][0]["mountPath"] == "/app/models"


def test_config_and_secret_are_injected_as_environment() -> None:
    config = one("ConfigMap")["data"]
    assert {"MODEL_STAGE", "LOG_LEVEL", "PORT"} <= config.keys()
    secret = one("Secret")
    assert secret["stringData"]["RELOAD_TOKEN"] == "", "never commit a real token"
    container = one("Deployment")["spec"]["template"]["spec"]["containers"][0]
    assert container["envFrom"][0]["configMapRef"]["name"] == (
        one("ConfigMap")["metadata"]["name"]
    )
    ref = container["env"][0]["valueFrom"]["secretKeyRef"]
    assert ref["name"] == secret["metadata"]["name"] and ref["key"] == "RELOAD_TOKEN"


def test_disruption_budget_selects_the_deployment() -> None:
    pdb = one("PodDisruptionBudget")["spec"]
    labels = one("Deployment")["spec"]["template"]["metadata"]["labels"]
    assert pdb["selector"]["matchLabels"].items() <= labels.items()
    assert pdb["maxUnavailable"] == 1


@pytest.mark.parametrize("kind", ["Deployment", "CronJob"])
def test_pod_security_is_restricted_compatible(kind: str) -> None:
    doc = one(kind)
    spec = (
        doc["spec"]["template"]["spec"]
        if kind == "Deployment"
        else (doc["spec"]["jobTemplate"]["spec"]["template"]["spec"])
    )
    assert spec["securityContext"]["runAsNonRoot"] is True
    assert spec["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
    assert spec["automountServiceAccountToken"] is False
    for container in spec["containers"]:
        ctx = container["securityContext"]
        assert ctx["allowPrivilegeEscalation"] is False
        assert ctx["readOnlyRootFilesystem"] is True
        assert ctx["capabilities"]["drop"] == ["ALL"]


def test_volumes_reference_the_pvc_and_api_mounts_it_read_only() -> None:
    pvc = one("PersistentVolumeClaim")["metadata"]["name"]
    pod = one("Deployment")["spec"]["template"]["spec"]
    claims = {
        v["persistentVolumeClaim"]["claimName"]
        for v in pod["volumes"]
        if "persistentVolumeClaim" in v
    }
    assert claims == {pvc}
    mount = next(
        m for m in pod["containers"][0]["volumeMounts"] if m["name"] == "models"
    )
    assert mount["readOnly"] is True


def test_cronjob_runs_the_three_stages_and_never_overlaps() -> None:
    cron = one("CronJob")["spec"]
    assert cron["concurrencyPolicy"] == "Forbid"
    script = cron["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]["args"][0]
    stages = [
        m.group(0)
        for m in re.finditer(r"src\.(etl_fetch|features|train_forecast)", script)
    ]
    assert stages == ["src.etl_fetch", "src.features", "src.train_forecast"]


def test_kustomization_pins_image_to_a_commit_sha() -> None:
    kustomization = yaml.safe_load((K8S / "kustomization.yaml").read_text())
    image = kustomization["images"][0]
    assert re.fullmatch(r"[0-9a-f]{40}", image["newTag"])
    listed = set(kustomization["resources"])
    assert listed == {p.name for p in FILES}


def test_training_job_keeps_data_on_the_models_claim() -> None:
    spec = one("CronJob")["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    env = {e["name"]: e["value"] for e in spec["containers"][0]["env"]}
    assert all(v.startswith("/app/models/data/") for v in env.values())
    assert [v["name"] for v in spec["volumes"]] == ["models"]
