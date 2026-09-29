#!/usr/bin/env python3
"""
Deploy Day 3 bootcamp pods on RunPod.

Creates GPU pods and (when a git deploy key is provided) clones the bootcamp
repo into /workspace/aisb. Participants connect via Jupyter (browser).

Usage:
    python deploy_runpod.py --git-private-key KEY           # Deploy 1 pod
    python deploy_runpod.py --count 15 --git-private-key KEY
    python deploy_runpod.py --list                          # List running pods
    python deploy_runpod.py --stop-all                      # Terminate all bootcamp pods
    python deploy_runpod.py --stop <pod_id>                 # Terminate one pod

Verifying content on a provisioned fleet:
    python deploy_runpod.py --test-solutions --day 3        # every 3.x section
    python deploy_runpod.py --test-solutions --all          # every section
    python deploy_runpod.py --test-solutions --day 3 --pod <pod_id>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from dotenv import dotenv_values

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Our own prints go through Python's stdout, but the pod-side commands we drive
# (git, pip, model downloads) inherit the raw fd and write to it directly. When
# output is redirected to a file or a pipe -- `... | tee deploy.log`, or a nohup
# background run -- Python block-buffers its half while the subprocesses do not,
# so progress messages arrive minutes late and interleaved out of order. Line
# buffering keeps the two streams in sync.
sys.stdout.reconfigure(line_buffering=True)

# This script lives at <repo>/runpod_setup/deploy_runpod.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
DOTENV_PATH = PROJECT_ROOT / ".env"

env_values = dotenv_values(DOTENV_PATH)
if env_values.get("RUNPOD_API_KEY"):
    os.environ["RUNPOD_API_KEY"] = env_values["RUNPOD_API_KEY"]

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
API_URL = "https://api.runpod.io/graphql"

POD_NAME_PREFIX = "aisb-bootcamp"
GPU_TYPE_IDS = [
    # Mid-price, strong price/performance
    "NVIDIA RTX 6000 Ada Generation",
    # "NVIDIA L40S", 
    "NVIDIA L40",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA A100-SXM4-40GB",

    # Expensive, high-performance
    "NVIDIA H100 PCIe",
    "NVIDIA H100 NVL",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H200 NVL",

    # Cheap fallbacks
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA RTX 5000 Ada Generation",
    "NVIDIA L4",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 3090 Ti",
    "NVIDIA RTX A5000",
]
DOCKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CONTAINER_DISK_GB = 100
VOLUME_GB = 100
VOLUME_MOUNT = "/workspace"
PORTS = "8888/http,22/tcp"
JUPYTER_PASSWORD = "aisbbootcamp"

# Repo cloned on each pod when --git-private-key is supplied
GIT_REPO_URL = "git@github.com:AI-Security-Bootcamp/aisb.git"
GIT_CLONE_DIR = "/workspace/aisb"

# Section folder holding setup/setup_pod.sh inside the cloned repo. This is the
# full setup used for every GPU day (3+): it installs Day 3's requirements,
# verifies the GPU, and pre-downloads the model cache.
SETUP_SECTION_DIR = "3.1-tokenization"

# Notebook/test tooling every day needs, regardless of its other deps.
NOTEBOOK_TOOLING = "jupyterlab ipywidgets ipykernel ipytest"

# Pod-side script that pre-installs the VS Code *server* extensions
# (Python/Pylance/Jupyter) into ~/.vscode-server, so Remote-SSH participants do
# not have to sit through the "Install in SSH" prompt. Repo-relative: the full
# profile runs it from inside setup_pod.sh, so only the lightweight profile
# below has to invoke it itself.
VSCODE_EXT_SCRIPT = f"{SETUP_SECTION_DIR}/setup/install_vscode_extensions.sh"

# Lightweight, API-only setup profiles for the days that need neither a GPU nor
# any pre-downloaded model weights. Each entry lists the pip commands to run on
# the pod (relative paths resolve inside the cloned repo). Days not listed here
# fall through to the full setup_pod.sh above.
#
# Day 1 is genuinely light (OpenRouter + small tokenizers). Day 2 is not a
# subset of the full setup: control-arena's dependency stack conflicts with the
# datasets pin in the main requirements, so it gets installed from its own
# requirements file. On RunPod (system Python, no venv) control-arena also needs
# blinker force-installed first -- see the header of requirements-control-arena.txt.
LIGHT_SETUP_DAYS: dict[str, list[str]] = {
    "1": [
        "pip install --upgrade pip",
        "pip install -r 1.1-llm-internals/requirements.txt",
        f"pip install {NOTEBOOK_TOOLING}",
    ],
    "2": [
        "pip install --upgrade pip",
        # control-arena pulls agentdojo -> flask -> blinker>=1.9; the apt-managed
        # blinker 1.4 cannot be uninstalled by pip, so force a newer copy ahead
        # of it (harmless on a pod that already has a recent blinker).
        "pip install --ignore-installed blinker",
        "pip install -r requirements-control-arena.txt",
        f"pip install {NOTEBOOK_TOOLING}",
    ],
}


def light_setup_commands(days: list[str] | None) -> list[str] | None:
    """Return the pip commands for a lightweight day, or None for full setup.

    Only a request for a *single* light day (exactly `--day 1` or `--day 2`)
    takes the lightweight path. Anything else -- no day given, `--all`, any GPU
    day, or a mix of days -- returns None so the caller runs the full setup.
    Days 1 and 2 in particular must not share a pod: their dependency stacks
    conflict, which is why a mixed request deliberately falls through to full
    setup rather than trying to install both.
    """
    if not days or len(days) != 1:
        return None
    return LIGHT_SETUP_DAYS.get(days[0])


# GPU days whose sections need packages beyond the base setup_pod.sh install (it
# covers only 3.1-tokenization/requirements.txt + the model cache). Keyed by day;
# each value is that day's own requirements.txt (repo-relative), which is the
# source of truth for its deps -- new packages are picked up automatically, no
# hand-maintained list to fall out of date. Installed on top of the base setup
# at provision time and by --test-solutions -- the GPU-day analogue of
# LIGHT_SETUP_DAYS. The per-day file is comprehensive (it re-lists torch etc.,
# already satisfied on the pod, so those are no-ops).
#
# NOTE on Day 5: its file pins datasets==3.6.0 (the fine-tuning pin, which
# CONFLICTS with the datasets>=4.4.2 control-arena wants -- never co-locate day 5
# with day 2) and vllm==0.6.3. vLLM has no wheel on some hosts and must never
# abort provisioning, so gpu_extra_setup_commands installs everything EXCEPT the
# vllm line normally, then vllm best-effort (see requirements-vllm.txt).
GPU_DAY_REQUIREMENTS: dict[str, str] = {
    "5": "5.1-model-editing/requirements.txt",
    "6": "6.1-adversarial-vision/requirements.txt",
}


def gpu_extra_setup_commands(days: list[str] | None) -> list[str] | None:
    """Install a single GPU day's requirements.txt on top of the base setup.

    Returns the pip commands, or None for anything but exactly one day that has
    a requirements file in GPU_DAY_REQUIREMENTS. Any `vllm` line is installed
    best-effort (it can fail to build on some hosts and must not abort the run);
    everything else in the file is a hard install.
    """
    if not days or len(days) != 1:
        return None
    req = GPU_DAY_REQUIREMENTS.get(days[0])
    if req is None:
        return None
    return [
        # Everything except any vllm line -- a hard install (aborts on failure).
        f"grep -ivE '^[[:space:]]*vllm' {req} | pip install -r /dev/stdin",
        # The vllm line, if present, best-effort so a missing wheel never fails
        # provisioning (only the optional Section 5.4 needs it).
        f"vllm_req=$(grep -iE '^[[:space:]]*vllm' {req} || true); "
        f"[ -n \"$vllm_req\" ] && (pip install \"$vllm_req\" || "
        f"echo 'WARNING: vllm install failed (optional; only Section 5.4 needs it)') || true",
    ]

# How long to wait for sshd inside a pod. A host that already has DOCKER_IMAGE
# cached is reachable in well under a minute; one that has to pull the ~20GB
# devel image can take the better part of ten. Budget for the cold case --
# giving up early skips the clone and setup phases on an otherwise fine pod.
SSH_WAIT_TIMEOUT = 900

# How long to allow for the repo clone/pull. The repo is small (~50 MB .git) and
# clones in seconds on a healthy host, but a degraded network path to GitHub can
# stall; give some headroom, then fail that pod gracefully rather than hang.
CLONE_TIMEOUT = 300

# ─────────────────────────────────────────────────────────────────────────────
# GraphQL helpers
# ─────────────────────────────────────────────────────────────────────────────

# The RunPod API intermittently returns an empty body, an HTML error page, or
# a 5xx -- especially while a large fleet is being scheduled. Those are
# transient: retrying the same request a few seconds later almost always
# succeeds. Treat them separately from GraphQL-level errors (a bad query or an
# unavailable GPU type), which will fail identically no matter how often we ask.
API_RETRIES = 5
API_RETRY_DELAY = 5


def graphql(query: str, variables: dict | None = None) -> dict:
    import subprocess, tempfile

    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    url = f"{API_URL}?api_key={API_KEY}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        tmp = f.name

    try:
        for attempt in range(1, API_RETRIES + 1):
            body = err = ""
            try:
                result = subprocess.run(
                    ["curl", "-s", "--show-error", "-X", "POST", url,
                     "-H", "Content-Type: application/json",
                     "-d", f"@{tmp}"],
                    capture_output=True, text=True, timeout=30,
                )
                body, err = result.stdout.strip(), result.stderr.strip()
                resp = json.loads(body)
            except Exception as e:
                # Empty body, HTML error page, curl transport failure, timeout.
                # Echo whatever we did get -- an HTML 502, a rate-limit message,
                # or curl's own "Could not resolve host" is far more diagnostic
                # than the bare "Expecting value: line 1 column 1" decode error.
                if body:
                    detail = f" (body: {body[:200]!r})"
                elif err:
                    detail = f" (curl: {err[:200]})"
                else:
                    detail = " (empty response)"
                if attempt == API_RETRIES:
                    print(f"API error after {API_RETRIES} attempts: {e}{detail}", file=sys.stderr)
                    sys.exit(1)
                print(
                    f"    API error ({e}){detail}; retrying in {API_RETRY_DELAY}s "
                    f"[{attempt}/{API_RETRIES}]",
                    file=sys.stderr, flush=True,
                )
                time.sleep(API_RETRY_DELAY)
                continue

            if "errors" in resp:
                # A real GraphQL error: report and stop. create_pod() relies on
                # this SystemExit to fall through to the next GPU type.
                print(f"GraphQL errors: {json.dumps(resp['errors'], indent=2)}", file=sys.stderr)
                sys.exit(1)

            return resp["data"]
    finally:
        os.unlink(tmp)

    raise AssertionError("unreachable")  # pragma: no cover


# ─────────────────────────────────────────────────────────────────────────────
# Pod operations
# ─────────────────────────────────────────────────────────────────────────────

def create_pod(name: str, gpu_type_id: str, ssh_public_key: str | None = None) -> dict:
    mutation = """
    mutation createPod($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        name
        desiredStatus
        imageName
        machine { podHostId maintenanceStart maintenanceEnd maintenanceNote }
      }
    }
    """
    variables = {
        "input": {
            "cloudType": "ALL",
            "gpuCount": 1,
            "volumeInGb": VOLUME_GB,
            "containerDiskInGb": CONTAINER_DISK_GB,
            "minVcpuCount": 4,
            "minMemoryInGb": 16,
            "gpuTypeId": gpu_type_id,
            "name": name,
            "imageName": DOCKER_IMAGE,
            "ports": PORTS,
            "volumeMountPath": VOLUME_MOUNT,
            "env": [
                {"key": "JUPYTER_PASSWORD", "value": JUPYTER_PASSWORD},
                {"key": "JUPYTER_TOKEN", "value": JUPYTER_PASSWORD},
                {"key": "HF_HOME", "value": "/workspace/model-cache"},
                {"key": "TRANSFORMERS_CACHE", "value": "/workspace/model-cache"},
                # Injected by the RunPod image into root's authorized_keys.
                *([{"key": "PUBLIC_KEY", "value": ssh_public_key}] if ssh_public_key else []),
            ],
        }
    }
    data = graphql(mutation, variables)
    return data["podFindAndDeployOnDemand"]


def pod_maintenance(pod: dict) -> tuple[str, str] | None:
    """Return (start, note) if the pod's host has a maintenance window, else None."""
    machine = pod.get("machine") or {}
    start = machine.get("maintenanceStart")
    if not start:
        return None
    return start, (machine.get("maintenanceNote") or "").strip()


# How long to wait for a pod's public port-22 mapping to appear. Hosts that
# publish one do so within seconds of the pod being scheduled; hosts that never
# will (see wait_for_public_ssh_port) stay empty indefinitely, so this only has
# to outlast normal scheduling jitter.
SSH_PORT_PUBLISH_TIMEOUT = 120


def wait_for_public_ssh_port(pod_id: str, timeout: int = SSH_PORT_PUBLISH_TIMEOUT) -> tuple[str, int] | None:
    """Poll until the pod publishes a public port-22 mapping.

    Distinct from wait_for_ssh, which waits for sshd *inside* a reachable pod.
    Some hosts never publish a public TCP mapping for port 22 at all -- the pod
    runs fine and Jupyter works over the HTTP proxy, but there is no address to
    SSH to, so the clone and setup phases can never run. Waiting on sshd in that
    state burns the full SSH_WAIT_TIMEOUT and still fails, so detect it here.
    """
    start = time.time()
    while time.time() - start < timeout:
        info = get_ssh_info(pod_id)
        if info:
            return info
        time.sleep(10)
    return None


def create_pod_on_healthy_host(
    name: str,
    gpu_type_ids: list[str],
    ssh_public_key: str | None = None,
    attempts: int = 3,
    allow_maintenance: bool = False,
    require_ssh: bool = False,
) -> dict | None:
    """Create a pod, rejecting any that lands on an unusable host.

    Two host-level problems are caught here: a scheduled maintenance window, and
    a host that never publishes a public port-22 mapping (require_ssh).

    RunPod schedules onto whichever machine has free capacity -- and a machine
    being drained for maintenance has plenty, so a naive retry lands on the very
    host you are trying to escape. There is no API to exclude a host, so instead
    we check the machine's maintenance window on arrival, terminate the pod if it
    is flagged, and try again. Cycling through gpu_type_ids is what actually
    moves us: different GPU models live on different machines, so changing the
    request is more likely to change the host than repeating it.

    Returns the pod, or None if every attempt landed on a flagged host or no GPU
    was available at all.
    """
    flagged_hosts: set[str] = set()
    for attempt in range(1, attempts + 1):
        for gpu in gpu_type_ids:
            try:
                pod = create_pod(name, gpu, ssh_public_key=ssh_public_key)
            except SystemExit:
                print(f"    {gpu} unavailable, trying next...", file=sys.stderr)
                continue

            maint = pod_maintenance(pod)
            if maint and not allow_maintenance:
                start, note = maint
                host = (pod.get("machine") or {}).get("podHostId", "?")
                flagged_hosts.add(host)
                print(
                    f"    {gpu} landed on host {host} with maintenance from {start}"
                    f" -- discarding [attempt {attempt}/{attempts}]\n"
                    f"      {note[:100]}",
                    file=sys.stderr,
                )
                stop_pod(pod["id"])
                continue

            # The pod is on an acceptable host; make sure we can actually reach
            # it before accepting it, otherwise clone and setup can never run.
            if require_ssh and not wait_for_public_ssh_port(pod["id"]):
                host = (pod.get("machine") or {}).get("podHostId", "?")
                flagged_hosts.add(host)
                print(
                    f"    {gpu} landed on host {host}, which published no public "
                    f"port 22 within {SSH_PORT_PUBLISH_TIMEOUT}s -- discarding "
                    f"[attempt {attempt}/{attempts}]",
                    file=sys.stderr,
                )
                stop_pod(pod["id"])
                continue

            suffix = ""
            if maint:
                suffix = f" (WARNING: host has maintenance scheduled from {maint[0]})"
            print(f"    Got {gpu}: {pod['id']}{suffix}")
            return pod

    if flagged_hosts:
        print(
            f"    Every attempt for '{name}' landed on an unusable host "
            f"({', '.join(sorted(flagged_hosts))}). Try again later, or create it "
            f"with --gpu <type> to target a different machine.",
            file=sys.stderr,
        )
    return None


def list_pods() -> list[dict]:
    query = """
    query {
      myself {
        pods {
          id
          name
          desiredStatus
          machine { podHostId maintenanceStart maintenanceEnd maintenanceNote }
          runtime {
            uptimeInSeconds
            ports { ip isIpPublic privatePort publicPort type }
          }
        }
      }
    }
    """
    return graphql(query)["myself"]["pods"]


def stop_pod(pod_id: str) -> None:
    mutation = """
    mutation terminatePod($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
    """
    graphql(mutation, {"input": {"podId": pod_id}})
    print(f"  Terminated pod {pod_id}")


def get_pod(pod_id: str) -> dict:
    query = """
    query getPod($input: PodFilter!) {
      pod(input: $input) {
        id
        name
        desiredStatus
        runtime {
          uptimeInSeconds
          ports { ip isIpPublic privatePort publicPort type }
        }
      }
    }
    """
    return graphql(query, {"input": {"podId": pod_id}})["pod"]


def get_ssh_info(pod_id: str) -> tuple[str, int] | None:
    """Get (ip, port) for SSH from a pod's runtime info."""
    pod = get_pod(pod_id)
    if not pod or not pod.get("runtime"):
        return None
    for p in pod["runtime"].get("ports", []):
        if p["privatePort"] == 22 and p["isIpPublic"]:
            return p["ip"], p["publicPort"]
    return None


# Shared ssh options. Kept as a list of (option, value) pairs so we can render
# them both as argv (for subprocess) and as a flag string (for user-facing
# "here's how to SSH in" messages).
_SSH_BASE_OPTS: list[tuple[str, str]] = [
    ("StrictHostKeyChecking", "no"),
    ("PasswordAuthentication", "no"),
]
# IdentitiesOnly prevents ssh-agent from offering other keys first. Only
# relevant when an explicit -i key is passed.
_SSH_KEY_OPT: tuple[str, str] = ("IdentitiesOnly", "yes")


def _ssh_opts_argv(ssh_key: str | None, extra: list[tuple[str, str]] | None = None) -> list[str]:
    opts = list(_SSH_BASE_OPTS)
    if extra:
        opts += extra
    argv: list[str] = []
    for k, v in opts:
        argv += ["-o", f"{k}={v}"]
    if ssh_key:
        argv += ["-i", ssh_key, "-o", f"{_SSH_KEY_OPT[0]}={_SSH_KEY_OPT[1]}"]
    return argv


def _ssh_opts_flagstr(ssh_key_arg: str) -> str:
    """Render shared ssh options as a flag string for display to the user."""
    parts = [f"-o {k}={v}" for k, v in _SSH_BASE_OPTS]
    parts = [f"-i {ssh_key_arg}"] + parts + [f"-o {_SSH_KEY_OPT[0]}={_SSH_KEY_OPT[1]}"]
    return " ".join(parts)


def build_ssh_cmd(pod_id: str, ssh_key: str | None = None) -> list[str] | None:
    """Build the ssh command prefix (without the remote command) for a pod.

    Returns None if the pod's SSH endpoint is not yet available. Callers
    append the remote command before invoking subprocess.
    """
    info = get_ssh_info(pod_id)
    if not info:
        return None
    ip, port = info
    cmd = ["ssh"] + _ssh_opts_argv(ssh_key, extra=[("ConnectTimeout", "10")])
    cmd += [f"root@{ip}", "-p", str(port)]
    return cmd


def pod_exec(
    pod_id: str,
    command: str,
    timeout: int | None = 120,
    ssh_key: str | None = None,
    quiet: bool = False,
    out_file: str | None = None,
) -> int:
    """Execute a command on a pod via SSH.

    By default stdout/stderr are inherited from the parent so the caller sees
    progress in real time. Returns the remote command's exit code, or -1 if we
    couldn't reach the pod over SSH.

    Pass quiet=True to swallow both streams -- used by the boot-polling loop,
    where a failed connection is the expected case and printing every refusal
    just buries the real output.

    Pass out_file to redirect both streams (append) to a file instead of the
    console -- used by the parallel test runner, where interleaving several
    pods' live output into one terminal would be unreadable.
    """
    import subprocess
    ssh_cmd = build_ssh_cmd(pod_id, ssh_key=ssh_key)
    if ssh_cmd is None:
        msg = "    Could not find SSH connection info\n"
        if out_file:
            with open(out_file, "a") as fh:
                fh.write(msg)
        elif not quiet:
            print(msg, end="", file=sys.stderr)
        return -1
    if out_file is not None:
        with open(out_file, "ab") as fh:
            result = subprocess.run(
                ssh_cmd + [command], timeout=timeout,
                stdout=fh, stderr=subprocess.STDOUT,
            )
        return result.returncode
    result = subprocess.run(
        ssh_cmd + [command], timeout=timeout, capture_output=quiet
    )
    return result.returncode


def wait_for_ssh(pod_id: str, timeout: int = SSH_WAIT_TIMEOUT, ssh_key: str | None = None) -> bool:
    """Poll until sshd inside the pod accepts a connection.

    A pod publishes its port mappings as soon as it is *scheduled*, which is
    what wait_for_pod keys off -- but sshd only starts once the container image
    has been pulled and the container is actually running. On a host with no
    cached copy of DOCKER_IMAGE that pull takes several minutes, and every
    connection until then is refused. Those refusals are the normal boot path,
    not a failure, so poll quietly and give the pull enough room to finish.
    """
    start = time.time()
    attempt = 0
    while time.time() - start < timeout:
        if pod_exec(pod_id, "true", timeout=15, ssh_key=ssh_key, quiet=True) == 0:
            return True
        attempt += 1
        # Report every ~60s so a long image pull looks like progress, not a hang.
        if attempt % 6 == 0:
            print(f"    still booting ({int(time.time() - start)}s elapsed)...", flush=True)
        time.sleep(10)
    return False


def install_git_key(pod_id: str, key_path: str, ssh_key: str | None = None) -> bool:
    """Install a private SSH key on the pod and configure it for github.com.

    Writes the key to ~/.ssh/id_github, adds an SSH config entry binding it
    to github.com (IdentitiesOnly yes), and seeds known_hosts via ssh-keyscan
    to avoid interactive host-verification prompts.
    """
    with open(key_path, "r", encoding="utf-8") as f:
        key_content = f.read()
    if not key_content.endswith("\n"):
        key_content += "\n"

    # A single shell script executed on the pod. The key is passed via a
    # quoted heredoc ('GIT_KEY_EOF') so $-expansion and backslashes in the
    # key material are preserved literally.
    script = (
        "set -e\n"
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh\n"
        "cat > ~/.ssh/id_github <<'GIT_KEY_EOF'\n"
        f"{key_content}"
        "GIT_KEY_EOF\n"
        "chmod 600 ~/.ssh/id_github\n"
        # Replace any prior github.com block, then append a fresh one.
        "touch ~/.ssh/config && chmod 600 ~/.ssh/config\n"
        "cat >> ~/.ssh/config <<'SSH_CFG_EOF'\n"
        "Host github.com\n"
        "    HostName github.com\n"
        "    User git\n"
        "    IdentityFile ~/.ssh/id_github\n"
        "    IdentitiesOnly yes\n"
        "SSH_CFG_EOF\n"
        "ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null || true\n"
        "chmod 600 ~/.ssh/known_hosts\n"
    )
    if pod_exec(pod_id, script, timeout=60, ssh_key=ssh_key) != 0:
        print("    FAILED to install git key", file=sys.stderr)
        return False
    print("    Installed git SSH key for github.com")
    return True


def clone_repo(
    pod_id: str,
    repo_url: str = GIT_REPO_URL,
    dest: str = GIT_CLONE_DIR,
    ssh_key: str | None = None,
) -> bool:
    """Clone (or pull) the bootcamp repo on the pod using the configured git key."""
    import subprocess
    # If the repo already exists from a prior run, pull instead of clone so
    # re-running the deploy is idempotent. `--depth 1` keeps the clone to the
    # latest commit only: the pods never need history, and a shallow clone is a
    # fraction of the transfer, so a slow host is far less likely to time out.
    # Treat the repo as "present" only if it has a valid HEAD -- a clone that was
    # interrupted (e.g. a timed-out earlier run) leaves a partial .git with no
    # HEAD, and blindly pulling that fails. In that case wipe it and clone fresh,
    # so --provision self-heals a broken clone instead of erroring on pull.
    script = (
        "set -e\n"
        f"if [ -d {dest}/.git ] && git -C {dest} rev-parse HEAD >/dev/null 2>&1; then\n"
        f"  echo 'Repo already present at {dest}, pulling latest...'\n"
        f"  git -C {dest} pull --ff-only\n"
        "else\n"
        f"  echo 'No valid clone at {dest}; cloning fresh...'\n"
        f"  rm -rf {dest}\n"
        f"  git clone --depth 1 {repo_url} {dest}\n"
        "fi\n"
    )
    # A slow or stalled network path to GitHub can exceed the timeout. Catch it
    # here so one unlucky pod fails gracefully (and prints the resume hint via
    # provision_pods) instead of crashing the whole deploy with a traceback.
    try:
        rc = pod_exec(pod_id, script, timeout=CLONE_TIMEOUT, ssh_key=ssh_key)
    except subprocess.TimeoutExpired:
        print(
            f"    TIMED OUT cloning after {CLONE_TIMEOUT}s (slow network to "
            f"GitHub?). The pod is fine; retry with --provision, or terminate "
            f"and recreate to land on a different host.",
            file=sys.stderr,
        )
        return False
    if rc != 0:
        print("    FAILED to clone repo", file=sys.stderr)
        return False
    print(f"    Cloned {repo_url} -> {dest}")
    return True


def run_setup_pod(
    pod_id: str,
    repo_dir: str = GIT_CLONE_DIR,
    ssh_key: str | None = None,
    days: list[str] | None = None,
) -> bool:
    """Set up the pod after the repo is cloned.

    For a single API-only day (`--day 1` or `--day 2`) this installs just that
    day's Python deps -- no GPU check, no model downloads -- which finishes in
    about a minute. For every other request (a GPU day, `--all`, no `--day`, or
    a mix of days) it runs the full setup_pod.sh, which installs Day 3's deps,
    verifies the GPU, and downloads the model cache -- several minutes.

    pod_exec streams output so the user sees progress live.
    """
    light = light_setup_commands(days)
    if light is not None:
        label = f"lightweight setup (day {days[0]}, API-only)"
        script = (
            "set -e\n"
            f"cd {repo_dir}\n"
            + "\n".join(light)
            + "\n"
            # setup_pod.sh does not run on this path, so seed the VS Code
            # extensions here. Never fatal -- a pod is fully usable without them.
            + f"bash {VSCODE_EXT_SCRIPT} || true\n"
        )
    else:
        # Base GPU setup (Day 3 deps, GPU check, model cache), plus any extra
        # requirements a specific GPU day needs on top of it (e.g. Day 5).
        extra = gpu_extra_setup_commands(days) or []
        extra_lines = "".join(f"{c}\n" for c in extra)
        label = "setup/setup_pod.sh"
        if extra:
            label += f" + day {days[0]} extra deps"
        script = (
            "set -e\n"
            f"cd {repo_dir}/{SETUP_SECTION_DIR}\n"
            "bash setup/setup_pod.sh\n"
            f"cd {repo_dir}\n"
            f"{extra_lines}"
        )
    rc = pod_exec(pod_id, script, timeout=1800, ssh_key=ssh_key)
    if rc != 0:
        print(f"    FAILED to run {label} (exit {rc})", file=sys.stderr)
        return False
    print(f"    Ran {label}")
    return True


# Running the full solution suite means model downloads and fine-tuning runs, so
# it needs far more headroom than setup_pod.sh gets.
TEST_SOLUTIONS_TIMEOUT = 14400

# Build tooling that test_solutions.py itself needs (via aisb_utils/solution_parsing.py)
# to turn a *_solution.py into the filled-in answers file. These live only in the
# top-level requirements.txt build-tooling section, which no pod setup profile
# installs -- participants never run test_solutions.py -- so --test-solutions
# installs them itself, on whatever pod it targets. Pins match requirements.txt.
TEST_SOLUTIONS_BUILD_TOOLING = "libcst~=1.8.0 black~=25.1.0 termcolor~=3.1.0"


def write_pod_env(
    pod_id: str,
    repo_dir: str,
    ssh_key: str | None,
    openrouter_key: str | None = None,
    hf_token: str | None = None,
) -> bool:
    """Write a temporary {repo_dir}/.env with the secrets a test run needs.

    OpenRouter key: the API-only days (1, 2) call aisb_utils/env.load_dotenv(),
    which reads this .env. HF token: gated-model days (e.g. 5.4) need it in the
    environment for huggingface_hub -- run_test_solutions sources this .env
    before the run so HF_TOKEN is exported, not merely present on disk.

    We create the file just for the run and delete it afterwards (remove_pod_env)
    so no secret persists on the pod. Values pass through a quoted heredoc so the
    shell does not expand them; they travel inside the SSH command over the
    encrypted channel, acceptable for an instructor's own throwaway testing pod.
    """
    lines, names = [], []
    if openrouter_key:
        lines.append(f"OPENROUTER_API_KEY={openrouter_key}")
        names.append("OpenRouter key")
    if hf_token:
        lines.append(f"HF_TOKEN={hf_token}")
        names.append("HF token")
    if not lines:
        return True
    body = "".join(f"{line}\n" for line in lines)
    # A quoted delimiter ('EOF') means no shell/parameter expansion in the body.
    script = (
        f"cat > {repo_dir}/.env <<'AISB_ENV_EOF'\n"
        f"{body}"
        "AISB_ENV_EOF\n"
    )
    rc = pod_exec(pod_id, script, timeout=30, ssh_key=ssh_key, quiet=True)
    if rc != 0:
        print(f"    FAILED to write .env on pod (exit {rc})", file=sys.stderr)
        return False
    print(f"    Wrote temporary .env ({', '.join(names)})")
    return True


def remove_pod_env(pod_id: str, repo_dir: str, ssh_key: str | None) -> None:
    """Delete the temporary {repo_dir}/.env written by write_pod_env.

    Best-effort and always run in a finally, so the key does not outlive the
    test run even if it failed or timed out.
    """
    rc = pod_exec(pod_id, f"rm -f {repo_dir}/.env\n", timeout=30, ssh_key=ssh_key, quiet=True)
    if rc == 0:
        print("    Removed temporary .env")
    else:
        print(f"    WARNING: could not remove {repo_dir}/.env (exit {rc})", file=sys.stderr)


def run_test_solutions(
    pod_id: str,
    day_flags: list[str],
    repo_dir: str = GIT_CLONE_DIR,
    ssh_key: str | None = None,
    timeout: int = TEST_SOLUTIONS_TIMEOUT,
    openrouter_key: str | None = None,
    hf_token: str | None = None,
    days: list[str] | None = None,
    out_file: str | None = None,
    section_timeout: int | None = None,
) -> int:
    """Run aisb_utils/test_solutions.py on one pod; return its exit code.

    That script fills in every solution, executes it against the pod's real
    GPU and installed dependencies, and deletes the generated answers file
    whether the run passes or fails. Output streams back through pod_exec, so
    tracebacks appear in this console as they happen, and the exit code is 0
    only if every section it was asked to test ran clean.

    If openrouter_key is given, a temporary .env holding it is written before
    the run and removed afterwards (the API-only days need it).

    When `days` names a single API-only day (1 or 2), that day's lightweight
    dependencies are installed first, so --test-solutions works on any cloned
    pod without having been provisioned for that day. GPU days (3+) install
    heavy deps + model weights, so those are assumed already provisioned.

    Returns -1 if the pod was unreachable or the run exceeded `timeout`.
    """
    import subprocess

    secrets = bool(openrouter_key or hf_token)
    if secrets and not write_pod_env(
        pod_id, repo_dir, ssh_key, openrouter_key=openrouter_key, hf_token=hf_token
    ):
        return -1

    # Install the day's own deps so --test-solutions works on any cloned pod:
    # a single API-only day (1/2) gets its lightweight stack; a single GPU day
    # with extras (e.g. 5) gets those on top of whatever setup_pod.sh installed.
    # These lists are mutually exclusive per day, so concatenating is safe.
    setup_cmds = (light_setup_commands(days) or []) + (gpu_extra_setup_commands(days) or [])
    setup_lines = "".join(f"{cmd}\n" for cmd in setup_cmds)

    # Per-section timeout inside test_solutions.py (its own DEFAULT_TIMEOUT is
    # 1800s). The long fine-tuning sections (e.g. 5.4) can exceed that, so allow
    # overriding it. When raised, also widen the whole-run SSH cap below so the
    # outer pod_exec does not kill the run before the per-section budget is spent.
    timeout_flag = f" --timeout {section_timeout}" if section_timeout else ""
    if section_timeout:
        # Room for several sections at the raised budget, plus setup overhead.
        timeout = max(timeout, section_timeout * 6 + 600)

    script = (
        "set -e\n"
        f"cd {repo_dir}\n"
        f"{setup_lines}"
        # Ensure the build tooling test_solutions.py imports is present. Quiet and
        # idempotent: a no-op once installed, so it costs only a few seconds on
        # reruns and makes --test-solutions work regardless of the pod's setup profile.
        f"pip install -q {TEST_SOLUTIONS_BUILD_TOOLING}\n"
        # Export any secrets from the temporary .env (HF_TOKEN for gated models,
        # OpenRouter key) into the environment for the run. `set -a` marks all
        # subsequently-sourced vars for export; huggingface_hub reads HF_TOKEN.
        f"set -a; [ -f {repo_dir}/.env ] && . {repo_dir}/.env; set +a\n"
        f"python3 aisb_utils/test_solutions.py {' '.join(day_flags)}{timeout_flag}\n"
    )
    try:
        return pod_exec(pod_id, script, timeout=timeout, ssh_key=ssh_key, out_file=out_file)
    except subprocess.TimeoutExpired:
        msg = f"    TIMED OUT after {timeout}s\n"
        if out_file:
            with open(out_file, "a") as fh:
                fh.write(msg)
        else:
            print(msg, end="", file=sys.stderr)
        return -1
    finally:
        # Always clean up the .env, whether the run passed, failed, or timed out.
        if secrets:
            remove_pod_env(pod_id, repo_dir, ssh_key)


# Lines of a failing pod's captured log to replay in the parallel summary, so a
# failure is visible without opening the log file.
PARALLEL_FAIL_TAIL_LINES = 40


def _result_label(returncode: int) -> str:
    """Human-readable status for a run_test_solutions exit code."""
    if returncode == 0:
        return "PASS"
    if returncode == -1:
        return "unreachable/timeout"
    return f"FAIL (exit {returncode})"


def _tail(path: str, n: int) -> list[str]:
    """Return the last n lines of a text file (best effort)."""
    try:
        with open(path, errors="replace") as fh:
            return fh.readlines()[-n:]
    except OSError:
        return []


def test_solutions_on_pods(
    pods: list[dict],
    day_flags: list[str],
    ssh_key: str | None = None,
    openrouter_key: str | None = None,
    hf_token: str | None = None,
    days: list[str] | None = None,
    parallel: bool = False,
    section_timeout: int | None = None,
) -> int:
    """Run the solution suite across a fleet; return how many pods failed.

    Sequential by default: pods run one at a time so their long, verbose output
    streams live and every traceback is attributable to the pod above it.

    With parallel=True the pods run concurrently instead. Live streaming would
    be unreadable interleaved, so each pod's output is captured to its own log
    file and only a PASS/FAIL summary table (plus the tail of each failing log)
    is printed at the end. Best for the long GPU days, where the wall-clock
    saving is large and you mostly care about the per-pod verdict.
    """
    print(f"\n{'='*70}")
    mode = "concurrently" if parallel and len(pods) > 1 else "one at a time"
    print(f"  Running test_solutions.py {' '.join(day_flags)} on {len(pods)} pod(s) ({mode})")
    print(f"{'='*70}")

    if parallel and len(pods) > 1:
        return _test_solutions_parallel(pods, day_flags, ssh_key, openrouter_key, hf_token, days, section_timeout)

    failed: list[tuple[dict, int]] = []
    for pod in sorted(pods, key=lambda p: p["name"]):
        print(f"\n  --- {pod['name']} ({pod['id']})")
        returncode = run_test_solutions(
            pod["id"], day_flags, ssh_key=ssh_key, openrouter_key=openrouter_key,
            hf_token=hf_token, days=days, section_timeout=section_timeout,
        )
        if returncode != 0:
            failed.append((pod, returncode))

    print(f"\n{'='*70}")
    if failed:
        print(f"  test_solutions.py FAILED on {len(failed)}/{len(pods)} pod(s):")
        for pod, returncode in failed:
            reason = "unreachable or timed out" if returncode == -1 else f"exit {returncode}"
            print(f"    {pod['name']} ({pod['id']}) -- {reason}")
        print("\n  Each pod's per-section summary is above; scroll up for tracebacks.")
    else:
        print(f"  test_solutions.py passed on all {len(pods)} pod(s).")
    print(f"{'='*70}", flush=True)
    return len(failed)


def _test_solutions_parallel(
    pods: list[dict],
    day_flags: list[str],
    ssh_key: str | None,
    openrouter_key: str | None,
    hf_token: str | None,
    days: list[str] | None,
    section_timeout: int | None = None,
) -> int:
    """Run test_solutions.py on every pod at once, capturing per-pod logs.

    Returns how many pods failed. Each pod's stdout/stderr goes to its own file
    under a temp directory; when they all finish we print a summary table and
    replay the tail of every failing log.
    """
    import concurrent.futures
    import tempfile

    log_dir = tempfile.mkdtemp(prefix="aisb_test_solutions_")
    ordered = sorted(pods, key=lambda p: p["name"])
    log_paths = {p["id"]: os.path.join(log_dir, f"{p['name']}_{p['id']}.log") for p in ordered}

    print(f"  Per-pod logs (streaming live to disk):")
    for pod in ordered:
        print(f"    {pod['name']:<18} {log_paths[pod['id']]}")
    print(f"  Tail any of them, e.g.  tail -f {log_paths[ordered[0]['id']]}", flush=True)

    def _run(pod: dict) -> tuple[dict, int]:
        rc = run_test_solutions(
            pod["id"], day_flags, ssh_key=ssh_key, openrouter_key=openrouter_key,
            hf_token=hf_token, days=days, out_file=log_paths[pod["id"]],
            section_timeout=section_timeout,
        )
        return pod, rc

    results: dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ordered)) as executor:
        futures = [executor.submit(_run, pod) for pod in ordered]
        for future in concurrent.futures.as_completed(futures):
            pod, returncode = future.result()
            results[pod["id"]] = returncode
            print(f"  finished: {pod['name']} ({pod['id']}) -- {_result_label(returncode)}",
                  flush=True)

    failed = [(pod, results[pod["id"]]) for pod in ordered if results[pod["id"]] != 0]

    # Summary table.
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'NAME':<18} {'ID':<16} {'RESULT':<20} LOG")
    for pod in ordered:
        rc = results[pod["id"]]
        print(f"  {pod['name']:<18} {pod['id']:<16} {_result_label(rc):<20} {log_paths[pod['id']]}")

    # Replay the tail of each failing log so failures are visible inline.
    for pod, _ in failed:
        print(f"\n{'-'*70}")
        print(f"  tail of {pod['name']} ({pod['id']}):")
        print(f"{'-'*70}")
        for line in _tail(log_paths[pod["id"]], PARALLEL_FAIL_TAIL_LINES):
            print(f"  {line}", end="")

    print(f"\n{'='*70}")
    if failed:
        print(f"  test_solutions.py FAILED on {len(failed)}/{len(ordered)} pod(s). "
              f"Full logs in {log_dir}")
    else:
        print(f"  test_solutions.py passed on all {len(ordered)} pod(s). "
              f"Logs in {log_dir}")
    print(f"{'='*70}", flush=True)
    return len(failed)


def provision_pods(
    pods: list[dict],
    git_private_key: str | None,
    ssh_key: str | None = None,
    days: list[str] | None = None,
) -> None:
    """Clone the repo onto every pod, then run setup_pod.sh on every pod.

    Provisioning runs in two fleet-wide phases rather than pod-by-pod. Cloning
    is quick and is what makes a pod usable at all, so getting it done
    everywhere first means a failure or interruption during the slow phase 2
    leaves the whole fleet in the same recoverable state -- repo present, deps
    pending -- instead of some pods fully provisioned and the rest bare.

    Safe to re-run against pods that are already partly provisioned: the clone
    step pulls instead of cloning, and setup_pod.sh is itself idempotent.
    """
    print(f"\n{'='*70}")
    print(f"  PHASE 1/2: waiting for {len(pods)} pod(s) and cloning the repo")
    print(f"{'='*70}")

    cloned_pods = []
    unreachable_pods = []
    for pod in pods:
        ready_pod = wait_for_pod(pod["id"])
        print_connection_info(ready_pod, ssh_key=ssh_key)

        if not git_private_key:
            continue

        print(f"\n  Waiting for sshd on {pod['id']} (image pull can take minutes)...")
        if not wait_for_ssh(pod["id"], ssh_key=ssh_key):
            print(
                f"    SSH never came up within {SSH_WAIT_TIMEOUT}s; skipping clone/setup.",
                file=sys.stderr,
            )
            unreachable_pods.append(pod)
            continue
        print("    SSH ready!")

        if not install_git_key(pod["id"], git_private_key, ssh_key=ssh_key):
            continue
        if clone_repo(pod["id"], ssh_key=ssh_key):
            cloned_pods.append(pod)

    # Every reachable pod is now cloned, so re-fetch and print the same listing
    # that --list produces, sorted by pod name. Doing it here means the
    # instructor has the full handout table before the slow phase 2 output
    # scrolls it off the screen.
    print(f"\n{'='*70}")
    print("  Cloning complete -- pod list (same as --list)")
    print(f"{'='*70}")
    bootcamp_pods = [p for p in list_pods() if p["name"].startswith(POD_NAME_PREFIX)]
    print_pod_list(bootcamp_pods, ssh_key=ssh_key)

    if cloned_pods:
        light = light_setup_commands(days)
        print(f"\n{'='*70}")
        print(f"  PHASE 2/2: setting up {len(cloned_pods)} pod(s)")
        if light is not None:
            print(f"  (day {days[0]} API-only: pip installs, no GPU check or models; ~1 min per pod)")
        else:
            print("  (full setup: pip installs + model downloads; several minutes per pod)")
        print(f"{'='*70}")

        failed_setup = []
        for pod in cloned_pods:
            print(f"\n  --- {pod['name']} ({pod['id']})")
            if not run_setup_pod(pod["id"], ssh_key=ssh_key, days=days):
                failed_setup.append(pod)

        if failed_setup:
            names = ", ".join(p["name"] for p in failed_setup)
            print(
                f"\n  WARNING: setup failed on {len(failed_setup)} pod(s): {names}",
                file=sys.stderr,
            )
            if light is not None:
                retry = "; ".join(light)
                print(
                    "  Their repo clone is intact, so re-run setup on each with:\n"
                    f"    ssh <pod> 'cd {GIT_CLONE_DIR} && {retry}'",
                    file=sys.stderr,
                )
            else:
                print(
                    "  Their repo clone is intact, so re-run setup on each with:\n"
                    f"    ssh <pod> 'cd {GIT_CLONE_DIR}/{SETUP_SECTION_DIR} && bash setup/setup_pod.sh'",
                    file=sys.stderr,
                )
            unreachable_pods += failed_setup

    if unreachable_pods:
        # These pods exist and are billing, they just are not provisioned yet.
        # Hand back the exact command that finishes them without creating new
        # ones -- re-running the plain deploy would double the fleet.
        ids = " ".join(f"--pod {p['id']}" for p in unreachable_pods)
        key_flags = f" --ssh-key {ssh_key}" if ssh_key else ""
        git_flags = f" --git-private-key {git_private_key}" if git_private_key else ""
        day_flags = "".join(f" --day {d}" for d in days) if days else ""
        print(f"\n{'='*70}")
        print(f"  {len(unreachable_pods)} pod(s) still need provisioning. Resume with:")
        print(f"\n    python {Path(__file__).name} --provision {ids}{key_flags}{git_flags}{day_flags}\n")
        print(f"{'='*70}", flush=True)


def wait_for_pod(pod_id: str, timeout: int = 300) -> dict:
    print(f"  Waiting for pod {pod_id}...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        pod = get_pod(pod_id)
        if pod and pod.get("runtime") and pod["runtime"].get("ports"):
            if any(p["privatePort"] == 8888 for p in pod["runtime"]["ports"]):
                print(" running!")
                return pod
        print(".", end="", flush=True)
        time.sleep(5)
    print(" timeout!")
    return get_pod(pod_id)


def derive_public_key(private_key_path: str) -> str:
    """Return the OpenSSH-format public key matching a private key on disk.

    Prefers a sibling `<path>.pub` file (common convention), falling back to
    `ssh-keygen -y -f <path>`. The result is what we inject as PUBLIC_KEY on
    each pod so that our local ssh (using the private key) can authenticate.
    """

    import subprocess
    import shutil
    import stat
    import tempfile

    pub_path = private_key_path + ".pub"
    if os.path.isfile(pub_path):
        return Path(pub_path).read_text(encoding="utf-8").strip()

    # ssh-keygen refuses to read keys with "too open" permissions (common on
    # Windows where files inherit an "Authenticated Users" ACL). Copy the key
    # into a tempfile that inherits the user's restrictive %TEMP% ACL, and on
    # POSIX tighten it to 0600, so ssh-keygen accepts it.
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        shutil.copyfile(private_key_path, tmp_path)
        if os.name == "posix":
            os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)
        result = subprocess.run(
            ["ssh-keygen", "-y", "-f", tmp_path],
            capture_output=True, text=True, timeout=10,
        )
    finally:
        os.unlink(tmp_path)

    if result.returncode != 0:
        raise RuntimeError(f"ssh-keygen failed: {result.stderr.strip()}")
    return result.stdout.strip()


def print_connection_info(pod: dict, ssh_key: str | None = None) -> None:
    pod_id = pod["id"]
    # A pod that is still booting has runtime: null (not an empty object), so
    # `.get("runtime", {})` still hands back None -- guard explicitly or --list
    # crashes whenever any pod in the fleet is starting up.
    ports = (pod.get("runtime") or {}).get("ports") or []
    ssh_flags = _ssh_opts_flagstr(ssh_key or "<path-to-key>")

    jupyter_url = f"https://{pod_id}-8888.proxy.runpod.net"
    print(f"\n  Pod: {pod['name']} ({pod_id})")
    print(f"  Status: {pod.get('desiredStatus', 'unknown')}")
    print(f"  Jupyter:  {jupyter_url}")
    print(f"  Password: {JUPYTER_PASSWORD}")
    print(f"  Terminal: https://www.runpod.io/console/pods/{pod_id}/terminal")
    # A flagged host will go down and take the pod with it, so make that
    # impossible to miss in the handout listing.
    maint = pod_maintenance(pod)
    if maint:
        start, note = maint
        end = (pod.get("machine") or {}).get("maintenanceEnd") or "?"
        print(f"  *** MAINTENANCE: host down {start} -> {end}")
        if note:
            print(f"      {note[:100]}")
    for p in ports:
        if p["privatePort"] == 22 and p["isIpPublic"]:
            print(f"  SSH:      ssh root@{p['ip']} -p {p['publicPort']} {ssh_flags}")
            break


def pod_ssh_endpoint(pod: dict) -> tuple[str, int] | None:
    """Return (ip, public_port) for a pod's SSH mapping, from its dict.

    Reads the ports already present on the pod dict rather than re-querying the
    API (get_ssh_info does that), so a whole listing costs no extra calls. A pod
    that is still booting has no public port-22 mapping yet and returns None.
    """
    ports = (pod.get("runtime") or {}).get("ports") or []
    for p in ports:
        if p["privatePort"] == 22 and p["isIpPublic"]:
            return p["ip"], p["publicPort"]
    return None


def print_pod_table(pods: list[dict]) -> None:
    """Print an aligned Name / ID / IP / Port table for a set of pods.

    Compact, copy-friendly summary to close out a listing. Pods still booting
    (no public SSH mapping yet) show "-" for IP and Port.
    """
    rows = []
    for pod in sorted(pods, key=lambda p: p["name"]):
        endpoint = pod_ssh_endpoint(pod)
        ip, port = (endpoint[0], str(endpoint[1])) if endpoint else ("-", "-")
        rows.append((pod["name"], pod["id"], ip, port))

    headers = ("NAME", "ID", "IP", "PORT")
    # Column width = widest cell (header included) in that column.
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(f"\n{'='*70}")
    print("  Pod summary")
    print(f"{'='*70}")
    print(f"  {fmt(headers)}")
    print(f"  {fmt(tuple('-' * w for w in widths))}")
    for row in rows:
        print(f"  {fmt(row)}")


def print_pod_list(pods: list[dict], ssh_key: str | None = None) -> None:
    """Print connection info for a set of pods, ordered by pod name.

    The RunPod API returns pods in arbitrary order, so sort by name to get
    aisb-bootcamp-01, -02, ... That makes the output directly usable as the
    handout table mapping each participant pair to a pod. A compact
    Name/ID/IP/Port table is printed at the end for quick copy/paste.
    """
    print(f"\n{'='*70}")
    print(f"  Bootcamp Pods ({len(pods)})")
    print(f"{'='*70}")
    for pod in sorted(pods, key=lambda p: p["name"]):
        print_connection_info(pod, ssh_key=ssh_key)
    if pods:
        print_pod_table(pods)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deploy Day 3 bootcamp pods on RunPod")
    parser.add_argument("--count", type=int, default=1, help="Number of pods to create")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=f"Explicit pod name. Only valid with --count 1. Must start with "
             f"'{POD_NAME_PREFIX}' so --list/--stop-all still find it.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all bootcamp pods, ending with a Name/ID/IP/Port table.",
    )
    parser.add_argument("--stop-all", action="store_true", help="Stop all bootcamp pods")
    parser.add_argument("--stop", type=str, help="Stop a specific pod by ID")
    parser.add_argument(
        "--provision",
        action="store_true",
        help="Provision pods that already exist instead of creating new ones. "
             "Targets the pods given with --pod, or every bootcamp pod if none "
             "are given. Idempotent: clones are pulled, setup re-runs. Honors "
             "--day 1 / --day 2 for the lightweight API-only setup.",
    )
    parser.add_argument(
        "--pod",
        type=str,
        action="append",
        default=[],
        dest="pods",
        metavar="POD_ID",
        help="Pod ID to act on. Repeatable. Used with --provision.",
    )
    parser.add_argument(
        "--test-solutions",
        action="store_true",
        help="Run aisb_utils/test_solutions.py on existing pods: fill in every "
             "solution, execute it, delete the generated answers file, and "
             "report failures here. Requires --day N or --all. Targets the "
             "pods given with --pod, or every bootcamp pod if none are given.",
    )
    parser.add_argument(
        "--day",
        action="append",
        dest="days",
        metavar="N",
        help="With --test-solutions: the day to test, e.g. --day 3 covers every "
             "3.x folder (repeatable). When creating or provisioning pods, a "
             "single --day 1 or --day 2 runs the lightweight API-only setup "
             "(no GPU check, no model downloads); any other value or mix runs "
             "the full setup.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="With --test-solutions: test every section in the repo.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="ID",
        help="With --test-solutions: leave out a section (--skip 3.3) or a "
             "single exercise (--skip 3.3.5) on every target pod, and still "
             "count the rest as a pass. Repeatable. Passed straight through to "
             "test_solutions.py.",
    )
    parser.add_argument(
        "--remain",
        action="store_true",
        help="With --test-solutions: keep the generated *_answers.py files on "
             "the pod instead of deleting them, so a failing section can be "
             "edited and re-run by hand over SSH. Passed straight through to "
             "test_solutions.py. Those files contain every solution for the "
             "section, so do not leave them on a pod participants can reach.",
    )
    parser.add_argument(
        "--section-timeout",
        type=int,
        default=None,
        metavar="SECONDS",
        help="With --test-solutions: per-section timeout passed to "
             "test_solutions.py (its default is 1800s). Raise it for the long "
             "fine-tuning sections, e.g. --section-timeout 5400. The overall "
             "SSH cap is widened to match.",
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --test-solutions on more than one pod: run all target pods "
             "concurrently (the default). Each pod's output is captured to its "
             "own log file and a PASS/FAIL summary table is printed at the end "
             "(live streaming is off, since interleaved output is unreadable). "
             "Pass --no-parallel to run pods one at a time with live output. A "
             "single-pod run (--pod X) always streams live regardless.",
    )
    parser.add_argument("--gpu", type=str, default=GPU_TYPE_IDS[0], help="GPU type")
    parser.add_argument(
        "--allow-maintenance-host",
        action="store_true",
        help="Accept a pod even if its host has a maintenance window scheduled. "
             "By default such pods are discarded and creation is retried, since "
             "the host will go down and take the pod with it.",
    )
    parser.add_argument("--api-key", type=str, help="RunPod API key (or RUNPOD_API_KEY env)")
    parser.add_argument(
        "--openrouter-key",
        type=str,
        default=None,
        help="OpenRouter API key for --test-solutions on the API-only days (1, 2). "
             "A temporary .env holding it is written to the pod for the run and "
             "deleted afterwards. Falls back to the OPENROUTER_API_KEY env var.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for --test-solutions on days that load a gated "
             "model (e.g. Section 5.4). Written to the same temporary .env, "
             "sourced into the run so HF_TOKEN reaches huggingface_hub, and "
             "deleted afterwards. Falls back to the HF_TOKEN env var. The token's "
             "account must have accepted the gated model's license.",
    )
    parser.add_argument(
        "--ssh-key",
        type=str,
        default=None,
        help="Path to an SSH private key used for all pod SSH operations. The "
             "matching public key (from <path>.pub or ssh-keygen -y) is "
             "injected as PUBLIC_KEY into each pod at startup.",
    )
    parser.add_argument(
        "--git-private-key",
        type=str,
        default=None,
        help="Path to an SSH private key. Installed on each pod and used to "
             f"clone {GIT_REPO_URL} into {GIT_CLONE_DIR}.",
    )
    args = parser.parse_args()

    if args.git_private_key:
        if not os.path.isfile(args.git_private_key):
            print(f"Error: --git-private-key file not found: {args.git_private_key}", file=sys.stderr)
            sys.exit(1)

    ssh_key: str | None = None
    ssh_public_key: str | None = None
    if args.ssh_key:
        if not os.path.isfile(args.ssh_key):
            print(f"Error: --ssh-key file not found: {args.ssh_key}", file=sys.stderr)
            sys.exit(1)
        ssh_key = args.ssh_key
        try:
            ssh_public_key = derive_public_key(args.ssh_key)
        except Exception as e:
            print(f"Error: could not derive public key from {args.ssh_key}: {e}", file=sys.stderr)
            sys.exit(1)

    global API_KEY
    if args.api_key:
        API_KEY = args.api_key
    if not API_KEY:
        print("Error: Set RUNPOD_API_KEY env var or pass --api-key", file=sys.stderr)
        sys.exit(1)

    # ── List ──
    if args.list:
        pods = list_pods()
        bootcamp_pods = [p for p in pods if p["name"].startswith(POD_NAME_PREFIX)]
        if not bootcamp_pods:
            print("No bootcamp pods found.")
            return
        print_pod_list(bootcamp_pods, ssh_key=ssh_key)
        return

    # ── Stop ──
    if args.stop:
        stop_pod(args.stop)
        return

    if args.stop_all:
        pods = list_pods()
        bootcamp_pods = [p for p in pods if p["name"].startswith(POD_NAME_PREFIX)]
        if not bootcamp_pods:
            print("No bootcamp pods to stop.")
            return
        for pod in bootcamp_pods:
            stop_pod(pod["id"])
        print(f"Stopped {len(bootcamp_pods)} pods.")
        return

    # ── Test solutions on existing pods ──
    #
    # Run by hand once the fleet is provisioned, to confirm the content still
    # executes on the pods participants will actually get.
    if args.test_solutions:
        if bool(args.days) == bool(args.all):
            print(
                "Error: --test-solutions needs exactly one of --day N or --all",
                file=sys.stderr,
            )
            sys.exit(1)
        for day in args.days or []:
            # These are interpolated into a remote shell command.
            if not day.isdigit():
                print(f"Error: --day expects a number, got {day!r}", file=sys.stderr)
                sys.exit(1)
        for skip_id in args.skip:
            # Also interpolated into the remote command, so keep it to digits and dots.
            parts = skip_id.split(".")
            if len(parts) not in (2, 3) or not all(part.isdigit() for part in parts):
                print(
                    f"Error: --skip expects a section or exercise id like 3.3 or "
                    f"3.3.5, got {skip_id!r}",
                    file=sys.stderr,
                )
                sys.exit(1)
        day_flags = ["--all"] if args.all else [f"--day {day}" for day in args.days]
        day_flags += [f"--skip {skip_id}" for skip_id in args.skip]
        if args.remain:
            day_flags.append("--remain")

        if args.pods:
            target_pods = [get_pod(pod_id) for pod_id in args.pods]
            missing = [pid for pid, pod in zip(args.pods, target_pods) if not pod]
            if missing:
                print(f"Error: pod(s) not found: {', '.join(missing)}", file=sys.stderr)
                sys.exit(1)
        else:
            target_pods = [p for p in list_pods() if p["name"].startswith(POD_NAME_PREFIX)]
            if not target_pods:
                print("No bootcamp pods found to test.")
                return

        # The API-only days need an OpenRouter key; take it from the flag or env.
        openrouter_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
        # HF token for gated-model days (e.g. 5.4). Flag first, then env var.
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        # --all tests everything, so it can't map to a single day's light deps.
        setup_days = None if args.all else args.days
        failures = test_solutions_on_pods(
            target_pods, day_flags, ssh_key=ssh_key,
            openrouter_key=openrouter_key, hf_token=hf_token, days=setup_days,
            parallel=args.parallel, section_timeout=args.section_timeout,
        )
        sys.exit(1 if failures else 0)

    # ── Provision existing pods ──
    #
    # The resume path: pods were already created (by an earlier run that was
    # interrupted, or whose sshd was not up in time), so skip creation and go
    # straight to cloning + setup.
    if args.provision:
        if args.pods:
            target_pods = [get_pod(pod_id) for pod_id in args.pods]
            missing = [pid for pid, p in zip(args.pods, target_pods) if not p]
            if missing:
                print(f"Error: pod(s) not found: {', '.join(missing)}", file=sys.stderr)
                sys.exit(1)
        else:
            target_pods = [p for p in list_pods() if p["name"].startswith(POD_NAME_PREFIX)]
            if not target_pods:
                print("No bootcamp pods found to provision.")
                return
        target_pods.sort(key=lambda p: p["name"])

        names = ", ".join(p["name"] for p in target_pods)
        print(f"\n{'='*70}")
        print(f"  Provisioning {len(target_pods)} existing pod(s): {names}")
        print(f"{'='*70}")
        if not args.git_private_key:
            print(
                "  Note: --git-private-key not given, so pods will only be waited "
                "for and listed, not cloned or set up.",
                file=sys.stderr,
            )
        # --all (only meaningful for --test-solutions) forces full setup here.
        setup_days = None if args.all else args.days
        provision_pods(target_pods, args.git_private_key, ssh_key=ssh_key, days=setup_days)
        return

    # ── Create ──
    print(f"\n{'='*70}")
    print(f"  Deploying {args.count} pod(s)")
    print(f"  Image: {DOCKER_IMAGE}")
    print(f"  Jupyter password: {JUPYTER_PASSWORD}")
    print(f"{'='*70}\n")

    if args.name is not None:
        if args.count != 1:
            print("Error: --name only valid with --count 1", file=sys.stderr)
            sys.exit(1)
        if not args.name.startswith(POD_NAME_PREFIX):
            print(f"Error: --name must start with '{POD_NAME_PREFIX}'", file=sys.stderr)
            sys.exit(1)

    created_pods = []
    for i in range(args.count):
        if args.name is not None:
            name = args.name
        else:
            name = f"{POD_NAME_PREFIX}-{i+1:02d}" if args.count > 1 else POD_NAME_PREFIX
        print(f"  Creating pod '{name}'...")

        gpus = [args.gpu] if args.gpu != GPU_TYPE_IDS[0] else GPU_TYPE_IDS
        pod = create_pod_on_healthy_host(
            name,
            gpus,
            ssh_public_key=ssh_public_key,
            allow_maintenance=args.allow_maintenance_host,
            # Only meaningful when we intend to SSH in; without a key we never
            # connect, so a missing port-22 mapping does not matter.
            require_ssh=ssh_public_key is not None,
        )
        if pod is None:
            print(f"    FAILED -- no usable GPU available", file=sys.stderr)
            continue
        created_pods.append(pod)

    if not created_pods:
        print("\nNo pods created.", file=sys.stderr)
        sys.exit(1)

    # --all (only meaningful for --test-solutions) forces full setup here.
    setup_days = None if args.all else args.days
    provision_pods(created_pods, args.git_private_key, ssh_key=ssh_key, days=setup_days)

    print(f"""
{'='*70}
  PARTICIPANT INSTRUCTIONS
{'='*70}

  METHOD 1 -- Jupyter in the browser
  ----------------------------------------------------------------------
  1. Open Jupyter in your browser (URL above)
  2. Password: {JUPYTER_PASSWORD}
  3. The bootcamp repo is at {GIT_CLONE_DIR}

  METHOD 2 -- VS Code Remote-SSH
  ----------------------------------------------------------------------
  1. Install the client-side extensions in VS Code ON YOUR OWN MACHINE.
     These run in your local window, not on the pod, so they cannot be
     pre-installed for you:

       code --install-extension ms-python.python
       code --install-extension ms-toolsai.jupyter

     (Or use the Extensions view; opening this repo also offers them as
     workspace recommendations.)

  2. chmod 600 the key
       Unix:    chmod 600 <path_to_key>
       Windows: icacls <path_to_key> /reset
                icacls <path_to_key> /inheritance:r /grant:r "your_username:(R,W)"

  3. Ctrl+Shift+P -> "Remote-SSH: Add New SSH Host" -> paste the ssh line
     with this syntax:
       ssh root@<ip> -p <port> -i <path_to_key> -o StrictHostKeyChecking=no -o PasswordAuthentication=no -o IdentitiesOnly=yes

  4. "Remote-SSH: Connect to Host" -> platform Linux

  5. Open folder {GIT_CLONE_DIR}

  6. The Python, Pylance and Jupyter extensions are already installed on the
     pod, so no "Install in SSH" prompt should appear. If one does, accept it.

  7. Edit with .py and # %%
{'='*70}
""")


if __name__ == "__main__":
    main()
