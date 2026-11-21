"""The day-to-day scripts must never download or build anything.

Real docker, minikube and kubectl are replaced by stubs that log every call, so
the tests can assert what the scripts do, and above all what they never do.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ENV_SH = ROOT / "scripts" / "env.sh"
K8S_UP = ROOT / "scripts" / "k8s_up.sh"

STUB = """#!/usr/bin/env bash
echo "{name} $*" >> "$STUB_LOG"
{body}
"""

DOCKER = """
case "$1 $2" in
  "info "*) exit 0 ;;
  "image inspect")
    for arg in "${@:3}"; do
      [[ " $MISSING_IMAGES " == *" $arg "* ]] && exit 1
    done
    exit 0 ;;
esac
exit 0
"""

MINIKUBE = """
case "$1" in
  status) [[ -n "$MK_STATE" ]] && echo "$MK_STATE" || exit 85 ;;
  ip) echo 192.168.49.2 ;;
esac
exit 0
"""

KUBECTL = """
if [[ -n "$KUBECTL_FAIL_RE" && "$*" =~ $KUBECTL_FAIL_RE ]]; then exit 1; fi
exit 0
"""


@pytest.fixture
def sandbox(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (
        ("docker", DOCKER),
        ("minikube", MINIKUBE),
        ("kubectl", KUBECTL),
    ):
        stub = bin_dir / name
        stub.write_text(STUB.format(name=name, body=body))
        stub.chmod(0o755)
    (tmp_path / "env").write_text("RELOAD_TOKEN=x\n")
    (tmp_path / "hosts").write_text(
        "127.0.0.1 localhost\n192.168.49.2 btc-forecast.local\n"
    )
    return {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "STUB_LOG": str(tmp_path / "calls.log"),
        "ENV_FILE": str(tmp_path / "env"),
        "HOSTS_FILE": str(tmp_path / "hosts"),
        "MISSING_IMAGES": "",
        "MK_STATE": "Running",
        "KUBECTL_FAIL_RE": "",
    }


def run(
    script: Path, *args: str, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def calls(env: dict[str, str]) -> list[str]:
    log = Path(env["STUB_LOG"])
    return log.read_text().splitlines() if log.exists() else []


def assert_nothing_downloaded_or_built(log: list[str]) -> None:
    for line in log:
        words = line.split()
        assert "build" not in words, line
        assert "down" not in words, line
        assert "-v" not in words and "--volumes" not in words, line
        if "pull" in words:
            assert words[words.index("pull") + 1] == "never", line
        assert not (words[:2] == ["docker", "pull"]), line


class TestStart:
    def test_starts_existing_state_without_pulling_or_building(
        self, sandbox: dict[str, str]
    ) -> None:
        result = run(ENV_SH, "start", env=sandbox)
        assert result.returncode == 0, result.stderr
        log = calls(sandbox)
        up = next(c for c in log if " up " in c)
        assert "--pull never" in up and "--no-build" in up and " -d" in up
        assert not any(c.startswith("minikube start") for c in log)
        assert_nothing_downloaded_or_built(log)

    def test_restarts_a_stopped_cluster_from_its_saved_config(
        self, sandbox: dict[str, str]
    ) -> None:
        sandbox["MK_STATE"] = "Stopped"
        assert run(ENV_SH, "start", env=sandbox).returncode == 0
        starts = [c for c in calls(sandbox) if c.startswith("minikube start")]
        assert starts == ["minikube start"]  # no sizing flags, no --download-only

    def test_missing_image_stops_with_the_one_time_command(
        self, sandbox: dict[str, str]
    ) -> None:
        sandbox["MISSING_IMAGES"] = "btc-forecast-airflow:local"
        result = run(ENV_SH, "start", env=sandbox)
        assert result.returncode == 1
        assert (
            "make build" in result.stderr
            and "btc-forecast-airflow:local" in result.stderr
        )
        assert not any(" up " in c for c in calls(sandbox))

    def test_missing_cluster_points_to_the_one_time_setup(
        self, sandbox: dict[str, str]
    ) -> None:
        sandbox["MK_STATE"] = ""
        result = run(ENV_SH, "start", env=sandbox)
        assert result.returncode == 1 and "make k8s-up" in result.stderr
        assert not any(c.startswith("minikube start") for c in calls(sandbox))

    def test_missing_deployment_points_to_the_one_time_setup(
        self, sandbox: dict[str, str]
    ) -> None:
        sandbox["KUBECTL_FAIL_RE"] = "get deployment btc-forecast"
        result = run(ENV_SH, "start", env=sandbox)
        assert result.returncode == 1 and "make k8s-up" in result.stderr

    def test_kubernetes_can_be_skipped(self, sandbox: dict[str, str]) -> None:
        sandbox["K8S"] = "0"
        assert run(ENV_SH, "start", env=sandbox).returncode == 0
        assert not any(c.startswith(("minikube", "kubectl")) for c in calls(sandbox))

    def test_missing_env_file_is_reported(self, sandbox: dict[str, str]) -> None:
        sandbox["ENV_FILE"] = "/nonexistent/.env"
        result = run(ENV_SH, "start", env=sandbox)
        assert result.returncode == 1 and "cp .env.example .env" in result.stderr

    def test_stale_hosts_entry_gets_a_fix_command(
        self, sandbox: dict[str, str], tmp_path: Path
    ) -> None:
        (tmp_path / "hosts").write_text("10.0.0.9 btc-forecast.local\n")
        result = run(ENV_SH, "start", env=sandbox)
        assert result.returncode == 0
        assert "10.0.0.9" in result.stdout and "192.168.49.2" in result.stdout
        assert "sudo sed" in result.stdout

    def test_matching_hosts_entry_prints_no_note(self, sandbox: dict[str, str]) -> None:
        assert "note:" not in run(ENV_SH, "start", env=sandbox).stdout


class TestStopAndStatus:
    def test_stop_keeps_volumes_and_the_cluster(self, sandbox: dict[str, str]) -> None:
        assert run(ENV_SH, "stop", env=sandbox).returncode == 0
        log = calls(sandbox)
        assert any(" stop" in c and c.startswith("docker compose") for c in log)
        assert "minikube stop" in log
        assert_nothing_downloaded_or_built(log)
        assert not any("delete" in c for c in log)

    def test_status_is_read_only(self, sandbox: dict[str, str]) -> None:
        assert run(ENV_SH, "status", env=sandbox).returncode == 0
        assert_nothing_downloaded_or_built(calls(sandbox))

    def test_unknown_command_is_a_usage_error(self, sandbox: dict[str, str]) -> None:
        assert run(ENV_SH, "frobnicate", env=sandbox).returncode == 2


class TestDeployScript:
    def deploy_env(self, sandbox: dict[str, str]) -> dict[str, str]:
        return {**sandbox, "USE_MINIKUBE": "0"}

    def test_existing_model_skips_the_bootstrap_training_job(
        self, sandbox: dict[str, str]
    ) -> None:
        result = run(K8S_UP, env=self.deploy_env(sandbox))
        assert result.returncode == 0, result.stderr
        assert not any("create job" in c for c in calls(sandbox))
        assert "skipping the bootstrap" in result.stdout

    def test_fresh_volume_runs_the_bootstrap_training_job_once(
        self, sandbox: dict[str, str]
    ) -> None:
        env = {
            **self.deploy_env(sandbox),
            "KUBECTL_FAIL_RE": "test -e /app/models/CURRENT",
        }
        assert run(K8S_UP, env=env).returncode == 0
        jobs = [c for c in calls(sandbox) if "create job" in c]
        assert len(jobs) == 1 and "--from=cronjob/btc-train" in jobs[0]

    def test_installed_addons_are_not_enabled_again(
        self, sandbox: dict[str, str]
    ) -> None:
        assert run(K8S_UP, env=sandbox).returncode == 0
        log = calls(sandbox)
        assert not any("addons enable" in c for c in log)
        assert not any(c.startswith("minikube start") for c in log)

    def test_missing_addons_are_enabled_once_each(
        self, sandbox: dict[str, str]
    ) -> None:
        env = {
            **sandbox,
            "KUBECTL_FAIL_RE": (
                "get deployment (ingress-nginx-controller|metrics-server)"
            ),
        }
        assert run(K8S_UP, env=env).returncode == 0
        enabled = [c for c in calls(sandbox) if "addons enable" in c]
        assert sorted(enabled) == [
            "minikube addons enable ingress",
            "minikube addons enable metrics-server",
        ]

    def test_stopped_cluster_is_started_before_deploying(
        self, sandbox: dict[str, str]
    ) -> None:
        env = {**sandbox, "MK_STATE": "Stopped"}
        assert run(K8S_UP, env=env).returncode == 0
        assert any(c.startswith("minikube start") for c in calls(sandbox))
