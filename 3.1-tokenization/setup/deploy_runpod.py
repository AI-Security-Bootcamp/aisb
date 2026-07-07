#!/usr/bin/env python3
"""
Deploy Day 3 bootcamp pods on RunPod.

Creates GPU pods and (when a git deploy key is provided) clones the bootcamp
repo into /workspace/aisb-sg. Participants connect via Jupyter (browser).

Usage:
    python deploy_runpod.py --git-private-key KEY           # Deploy 1 pod
    python deploy_runpod.py --count 15 --git-private-key KEY
    python deploy_runpod.py --list                          # List running pods
    python deploy_runpod.py --stop-all                      # Terminate all bootcamp pods
    python deploy_runpod.py --stop <pod_id>                 # Terminate one pod
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

# This script lives at <repo>/3.1-tokenization/setup/deploy_runpod.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
DOTENV_PATH = PROJECT_ROOT / ".env"

env_values = dotenv_values(DOTENV_PATH)
if env_values["RUNPOD_API_KEY"]:
    os.environ["RUNPOD_API_KEY"] = env_values["RUNPOD_API_KEY"]

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
API_URL = "https://api.runpod.io/graphql"

POD_NAME_PREFIX = "day3-bootcamp"
GPU_TYPE_IDS = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA L4",
    "NVIDIA A40",
]
DOCKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CONTAINER_DISK_GB = 100
VOLUME_GB = 100
VOLUME_MOUNT = "/workspace"
PORTS = "8888/http,22/tcp"
JUPYTER_PASSWORD = "day3bootcamp"

# Repo cloned on each pod when --git-private-key is supplied
GIT_REPO_URL = "git@github.com:AI-Security-Bootcamp/aisb-sg.git"
GIT_CLONE_DIR = "/workspace/aisb-sg"

# ─────────────────────────────────────────────────────────────────────────────
# GraphQL helpers
# ─────────────────────────────────────────────────────────────────────────────

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
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", url,
             "-H", "Content-Type: application/json",
             "-d", f"@{tmp}"],
            capture_output=True, text=True, timeout=30,
        )
        resp = json.loads(result.stdout)
    except Exception as e:
        print(f"API error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        os.unlink(tmp)

    if "errors" in resp:
        print(f"GraphQL errors: {json.dumps(resp['errors'], indent=2)}", file=sys.stderr)
        sys.exit(1)

    return resp["data"]


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
        machine { podHostId }
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


def list_pods() -> list[dict]:
    query = """
    query {
      myself {
        pods {
          id
          name
          desiredStatus
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
) -> int:
    """Execute a command on a pod via SSH, streaming output live.

    stdout/stderr are inherited from the parent so the caller sees progress
    in real time. Returns the remote command's exit code, or -1 if we
    couldn't reach the pod over SSH.
    """
    import subprocess
    ssh_cmd = build_ssh_cmd(pod_id, ssh_key=ssh_key)
    if ssh_cmd is None:
        print("    Could not find SSH connection info", file=sys.stderr)
        return -1
    result = subprocess.run(ssh_cmd + [command], timeout=timeout)
    return result.returncode


def wait_for_ssh(pod_id: str, timeout: int = 180, ssh_key: str | None = None) -> bool:
    """Poll pod_exec until SSH accepts a simple command."""
    start = time.time()
    while time.time() - start < timeout:
        if pod_exec(pod_id, "true", timeout=15, ssh_key=ssh_key) == 0:
            return True
        time.sleep(5)
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
    # If the repo already exists from a prior run, pull instead of clone so
    # re-running the deploy is idempotent.
    script = (
        "set -e\n"
        f"if [ -d {dest}/.git ]; then\n"
        f"  echo 'Repo already present at {dest}, pulling latest...'\n"
        f"  git -C {dest} pull --ff-only\n"
        "else\n"
        f"  git clone {repo_url} {dest}\n"
        "fi\n"
    )
    if pod_exec(pod_id, script, timeout=180, ssh_key=ssh_key) != 0:
        print("    FAILED to clone repo", file=sys.stderr)
        return False
    print(f"    Cloned {repo_url} -> {dest}")
    return True


def run_setup_pod(
    pod_id: str,
    repo_dir: str = GIT_CLONE_DIR,
    ssh_key: str | None = None,
) -> bool:
    """Run the bootcamp's setup_pod.sh on the pod after the repo is cloned.

    This can take many minutes as it installs Python deps and downloads
    model weights; pod_exec streams output so the user sees progress live.
    """
    script = (
        "set -e\n"
        f"cd {repo_dir}/3.1-tokenization\n"
        "bash setup/setup_pod.sh\n"
    )
    rc = pod_exec(pod_id, script, timeout=1800, ssh_key=ssh_key)
    if rc != 0:
        print(f"    FAILED to run setup_pod.sh (exit {rc})", file=sys.stderr)
        return False
    print("    Ran setup/setup_pod.sh")
    return True


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
    ports = pod.get("runtime", {}).get("ports", [])
    ssh_flags = _ssh_opts_flagstr(ssh_key or "<path-to-key>")

    jupyter_url = f"https://{pod_id}-8888.proxy.runpod.net"
    print(f"\n  Pod: {pod['name']} ({pod_id})")
    print(f"  Status: {pod.get('desiredStatus', 'unknown')}")
    print(f"  Jupyter:  {jupyter_url}")
    print(f"  Password: {JUPYTER_PASSWORD}")
    print(f"  Terminal: https://www.runpod.io/console/pods/{pod_id}/terminal")
    for p in ports:
        if p["privatePort"] == 22 and p["isIpPublic"]:
            print(f"  SSH:      ssh root@{p['ip']} -p {p['publicPort']} {ssh_flags}")
            break


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
    parser.add_argument("--list", action="store_true", help="List all running pods")
    parser.add_argument("--stop-all", action="store_true", help="Stop all bootcamp pods")
    parser.add_argument("--stop", type=str, help="Stop a specific pod by ID")
    parser.add_argument("--gpu", type=str, default=GPU_TYPE_IDS[0], help="GPU type")
    parser.add_argument("--api-key", type=str, help="RunPod API key (or RUNPOD_API_KEY env)")
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
        print(f"\n{'='*70}")
        print(f"  Bootcamp Pods ({len(bootcamp_pods)})")
        print(f"{'='*70}")
        for pod in bootcamp_pods:
            print_connection_info(pod, ssh_key=ssh_key)
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

        pod = None
        for gpu in ([args.gpu] if args.gpu != GPU_TYPE_IDS[0] else GPU_TYPE_IDS):
            try:
                pod = create_pod(name, gpu, ssh_public_key=ssh_public_key)
                print(f"    Got {gpu}: {pod['id']}")
                break
            except SystemExit:
                print(f"    {gpu} unavailable, trying next...", file=sys.stderr)
                continue

        if pod is None:
            print(f"    FAILED -- no GPU available", file=sys.stderr)
            continue
        created_pods.append(pod)

    if not created_pods:
        print("\nNo pods created.", file=sys.stderr)
        sys.exit(1)

    # Wait for pods, upload files, print info
    print(f"\n{'='*70}")
    print(f"  Waiting for {len(created_pods)} pod(s)...")
    print(f"{'='*70}")

    for pod in created_pods:
        ready_pod = wait_for_pod(pod["id"])
        print_connection_info(ready_pod, ssh_key=ssh_key)

        if args.git_private_key:
            print(f"\n  Installing git SSH key on {pod['id']}...", end="", flush=True)
            if wait_for_ssh(pod["id"], timeout=180, ssh_key=ssh_key):
                print(" SSH ready!")
                if install_git_key(pod["id"], args.git_private_key, ssh_key=ssh_key):
                    if clone_repo(pod["id"], ssh_key=ssh_key):
                        print(f"\n{'='*70}")
                        print(f"  Running setup_pod.sh on {pod['name']} ({pod['id']})")
                        print("  (this may take several minutes)")
                        print(f"{'='*70}")
                        run_setup_pod(pod["id"], ssh_key=ssh_key)
            else:
                print(" SSH timed out; skipping git key install.", file=sys.stderr)

    print(f"""
{'='*70}
  PARTICIPANT INSTRUCTIONS
{'='*70}

  1. Open Jupyter in your browser (URL above)
  2. Password: {JUPYTER_PASSWORD}
  3. Open a terminal in Jupyter and run:
       cd {GIT_CLONE_DIR}/3.1-tokenization
       pip install transformers accelerate scikit-learn tqdm

  To iterate: edit .py files in Jupyter, then re-run from terminal or notebook.

  FIRST TIME: run the setup to download models (~5 min):
    cd {GIT_CLONE_DIR}/3.1-tokenization && bash setup/setup_pod.sh
{'='*70}
""")


if __name__ == "__main__":
    main()
