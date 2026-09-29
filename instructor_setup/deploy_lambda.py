#!/usr/bin/env python3
"""
Deploy a bootcamp instance on Lambda Cloud and lay out per-participant workspaces.

Companion to deploy_runpod.py, for the days that run on Lambda instead of
RunPod. It launches a single GPU instance, then creates one directory per
participant under the instance's home directory (test, user1 ... user20) and
places a clone of the bootcamp repo in each.

The repo is fetched from GitHub exactly once into a cache directory and then
copied into every participant directory. Twenty-one separate network clones of
the same commit would be slow and hammer GitHub for no benefit; the copies are
ordinary git working trees, so participants can still commit and diff locally.

Usage:
    python deploy_lambda.py --ssh-key ~/.ssh/aisb_key --git-private-key ~/.ssh/aisb_key
    python deploy_lambda.py --list                        # List bootcamp instances
    python deploy_lambda.py --provision --instance <id> --ssh-key K --git-private-key K
    python deploy_lambda.py --terminate <instance_id>
    python deploy_lambda.py --terminate-all
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

from dotenv import dotenv_values

# Our own prints go through Python's stdout, but the instance-side commands we
# drive (git, cp) inherit the raw fd and write to it directly. Line buffering
# keeps the two streams in order when output is piped or redirected to a log.
sys.stdout.reconfigure(line_buffering=True)

# This script lives at <repo>/instructor_setup/deploy_lambda.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"

env_values = dotenv_values(DOTENV_PATH)
if env_values.get("LAMBDA_API_KEY"):
    os.environ.setdefault("LAMBDA_API_KEY", env_values["LAMBDA_API_KEY"])

API_KEY = os.environ.get("LAMBDA_API_KEY", "")
API_URL = "https://cloud.lambda.ai/api/v1"

# Lambda's API sits behind Cloudflare, which rejects the default
# "Python-urllib/3.x" User-Agent with a 403 (error code 1010). Any descriptive
# agent string is accepted, so send one that identifies this script.
USER_AGENT = "aisb-deploy-lambda/1.0"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

INSTANCE_NAME_PREFIX = "aisb-bootcamp"
INSTANCE_TYPE = "gpu_1x_a10"

# Machine image to boot. A family name is resolved by Lambda to that family's
# current build in whichever region we land in, so it keeps working as Lambda
# republishes images; a pinned --image-id does not. Lambda Stack ships CUDA,
# Docker and the usual ML frameworks preinstalled.
IMAGE_FAMILY = "lambda-stack-24-04"

# Lambda instances always log in as `ubuntu`, unlike RunPod's root containers.
SSH_USER = "ubuntu"

# Name under which we register the deploy key in the Lambda account if the
# public key we were given is not there already.
SSH_KEY_NAME = "aisb-bootcamp-key"

# Repo cloned into every participant directory when --git-private-key is given.
GIT_REPO_URL = "git@github.com:AI-Security-Bootcamp/aisb.git"

# Directory name the repo lands in inside each participant directory, i.e.
# ~/user1/aisb.
CLONE_DIR_NAME = "aisb"

# Single fetch from GitHub is cached here and copied into each participant dir.
REPO_CACHE_DIR = "$HOME/.aisb-repo-cache"

# Participant directories: a scratch "test" workspace plus user1..userN.
DEFAULT_USER_COUNT = 20


def participant_dirs(count: int = DEFAULT_USER_COUNT) -> list[str]:
    """Names of the per-participant directories created in the home directory."""
    return ["test"] + [f"user{i}" for i in range(1, count + 1)]


# How long to wait for the instance to leave "booting" and report an IP.
BOOT_TIMEOUT = 900
# How long to wait for sshd once the instance is active. A freshly booted
# Lambda box answers within a minute; give cloud-init room beyond that.
SSH_WAIT_TIMEOUT = 600
# The one network clone plus 21 local copies of a ~50 MB repo.
PROVISION_TIMEOUT = 1800

# The API occasionally returns a 5xx or an empty body while capacity is being
# scheduled. Those are transient; a 4xx (bad request, no capacity, bad key) is
# not, and is surfaced immediately.
API_RETRIES = 5
API_RETRY_DELAY = 5


# ─────────────────────────────────────────────────────────────────────────────
# REST helpers
# ─────────────────────────────────────────────────────────────────────────────

def api(method: str, path: str, body: dict | None = None) -> dict:
    """Call the Lambda Cloud API and return the parsed `data` payload.

    Retries transient server-side failures; exits on a client error (4xx),
    printing the API's own error message, which is far more useful than a
    stack trace ("instance type not available in region", etc.).
    """
    url = f"{API_URL}/{path.lstrip('/')}"
    payload = json.dumps(body).encode() if body is not None else None

    for attempt in range(1, API_RETRIES + 1):
        request = urllib.request.Request(url, data=payload, method=method)
        request.add_header("Authorization", f"Bearer {API_KEY}")
        request.add_header("Content-Type", "application/json")
        request.add_header("User-Agent", USER_AGENT)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode())["data"]
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if 400 <= e.code < 500:
                print(f"API error {e.code} on {method} {path}: {detail}", file=sys.stderr)
                sys.exit(1)
            reason = f"HTTP {e.code}: {detail}"
        except Exception as e:  # transport error, timeout, malformed body
            reason = str(e)

        if attempt == API_RETRIES:
            print(f"API error after {API_RETRIES} attempts on {method} {path}: {reason}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"    API error ({reason}); retrying in {API_RETRY_DELAY}s "
              f"[{attempt}/{API_RETRIES}]", file=sys.stderr, flush=True)
        time.sleep(API_RETRY_DELAY)

    raise AssertionError("unreachable")  # pragma: no cover


# ─────────────────────────────────────────────────────────────────────────────
# Account resources: images, SSH keys, regions
# ─────────────────────────────────────────────────────────────────────────────

def normalize_image_id(image_id: str) -> str:
    """Return an image id in the dashed UUID form the API expects.

    Image ids are frequently passed around with the dashes stripped (32 hex
    characters). Re-insert them so either form works on the command line.
    """
    raw = image_id.replace("-", "")
    if len(raw) == 32 and all(c in "0123456789abcdefABCDEF" for c in raw):
        return f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"
    return image_id


def image_spec(image_id: str | None, image_family: str) -> dict:
    """Build the launch call's `image` field, warning if the choice is unknown.

    Prefers an explicit --image-id (normalized to dashed UUID form) over the
    family. The warning is advisory only: the image list is region-scoped and
    turns over as Lambda publishes new builds, so a miss is not proof the value
    is wrong, and a genuinely bad one fails fast with the API's own message.
    """
    images = api("GET", "images")
    if image_id:
        normalized = normalize_image_id(image_id)
        if not any(image["id"] == normalized for image in images):
            print(f"  WARNING: image id {normalized} is not in this account's "
                  f"image list ({len(images)} images); launch may be rejected.",
                  file=sys.stderr)
        return {"id": normalized}

    if not any(image["family"] == image_family for image in images):
        print(f"  WARNING: image family {image_family!r} is not in this account's "
              f"image list; launch may be rejected.", file=sys.stderr)
    return {"family": image_family}


def pick_region(instance_type: str, region: str | None = None) -> str:
    """Return a region with capacity for `instance_type`.

    Lambda's capacity moves between regions, so with no explicit --region we
    take the first one currently advertising free capacity rather than pinning
    a region that may be full.
    """
    types = api("GET", "instance-types")
    entry = types.get(instance_type)
    if entry is None:
        print(f"Error: unknown instance type {instance_type!r}", file=sys.stderr)
        sys.exit(1)
    available = [r["name"] for r in entry["regions_with_capacity_available"]]

    if region:
        if region not in available:
            print(f"Error: {instance_type} has no capacity in {region}. "
                  f"Available: {', '.join(available) or 'none'}", file=sys.stderr)
            sys.exit(1)
        return region

    if not available:
        print(f"Error: {instance_type} has no capacity in any region right now.",
              file=sys.stderr)
        sys.exit(1)
    price = entry["instance_type"]["price_cents_per_hour"] / 100
    print(f"  {instance_type} (${price:.2f}/hr) available in: {', '.join(available)}")
    return available[0]


def ensure_ssh_key(public_key: str) -> str:
    """Return the Lambda account's name for `public_key`, registering it if new.

    Lambda's launch call identifies keys by name, not by content, so we match on
    the key body (type + base64, ignoring the trailing comment) to find a key
    that is already there and only upload one when there is no match.
    """
    body = " ".join(public_key.split()[:2])
    for key in api("GET", "ssh-keys"):
        if " ".join(key["public_key"].split()[:2]) == body:
            print(f"  Using existing Lambda SSH key '{key['name']}'")
            return key["name"]

    created = api("POST", "ssh-keys", {"name": SSH_KEY_NAME, "public_key": public_key})
    print(f"  Registered new Lambda SSH key '{created['name']}'")
    return created["name"]


def derive_public_key(private_key_path: str) -> str:
    """Return the OpenSSH-format public key matching a private key on disk.

    Prefers a sibling `<path>.pub` file, falling back to `ssh-keygen -y`.
    """
    import shutil
    import stat
    import subprocess
    import tempfile

    pub_path = private_key_path + ".pub"
    if os.path.isfile(pub_path):
        return Path(pub_path).read_text(encoding="utf-8").strip()

    # ssh-keygen refuses keys with "too open" permissions, which is common when
    # a key has been copied around. Work from a 0600 temp copy instead.
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        shutil.copyfile(private_key_path, tmp_path)
        if os.name == "posix":
            os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)
        result = subprocess.run(["ssh-keygen", "-y", "-f", tmp_path],
                                capture_output=True, text=True, timeout=10)
    finally:
        os.unlink(tmp_path)

    if result.returncode != 0:
        raise RuntimeError(f"ssh-keygen failed: {result.stderr.strip()}")
    return result.stdout.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Instance operations
# ─────────────────────────────────────────────────────────────────────────────

def launch_instance(name: str, instance_type: str, region: str,
                    ssh_key_name: str, image: dict) -> str:
    """Launch one instance and return its id."""
    data = api("POST", "instance-operations/launch", {
        "region_name": region,
        "instance_type_name": instance_type,
        "ssh_key_names": [ssh_key_name],
        "file_system_names": [],
        "quantity": 1,
        "name": name,
        "image": image,
    })
    return data["instance_ids"][0]


def get_instance(instance_id: str) -> dict:
    return api("GET", f"instances/{instance_id}")


def list_instances() -> list[dict]:
    return api("GET", "instances")


def terminate_instance(instance_id: str) -> None:
    api("POST", "instance-operations/terminate", {"instance_ids": [instance_id]})
    print(f"  Terminated instance {instance_id}")


def wait_for_instance(instance_id: str, timeout: int = BOOT_TIMEOUT) -> dict:
    """Poll until the instance reports status 'active' with an IP address."""
    print(f"  Waiting for instance {instance_id}...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        instance = get_instance(instance_id)
        if instance.get("status") == "active" and instance.get("ip"):
            print(" active!")
            return instance
        if instance.get("status") in ("terminated", "unhealthy"):
            print(f" {instance['status']}!")
            return instance
        print(".", end="", flush=True)
        time.sleep(10)
    print(" timeout!")
    return get_instance(instance_id)


# ─────────────────────────────────────────────────────────────────────────────
# SSH helpers
# ─────────────────────────────────────────────────────────────────────────────

# Shared ssh options, kept as (option, value) pairs so they can be rendered both
# as argv (for subprocess) and as a flag string (for the "here's how to SSH in"
# lines we hand to participants).
_SSH_BASE_OPTS: list[tuple[str, str]] = [
    ("StrictHostKeyChecking", "no"),
    ("UserKnownHostsFile", "/dev/null"),
    ("PasswordAuthentication", "no"),
]
# Stops ssh-agent from offering other keys first. Only relevant with an -i key.
_SSH_KEY_OPT: tuple[str, str] = ("IdentitiesOnly", "yes")


def _ssh_opts_argv(ssh_key: str | None,
                   extra: list[tuple[str, str]] | None = None) -> list[str]:
    argv: list[str] = []
    for key, value in _SSH_BASE_OPTS + (extra or []):
        argv += ["-o", f"{key}={value}"]
    if ssh_key:
        argv += ["-i", ssh_key, "-o", f"{_SSH_KEY_OPT[0]}={_SSH_KEY_OPT[1]}"]
    return argv


def _ssh_opts_flagstr(ssh_key_arg: str) -> str:
    parts = [f"-i {ssh_key_arg}"]
    parts += [f"-o {k}={v}" for k, v in _SSH_BASE_OPTS]
    parts += [f"-o {_SSH_KEY_OPT[0]}={_SSH_KEY_OPT[1]}"]
    return " ".join(parts)


def instance_exec(ip: str, command: str, timeout: int | None = 120,
                  ssh_key: str | None = None, quiet: bool = False) -> int:
    """Run a command on the instance over SSH; return its exit code.

    Output is inherited from the parent by default so long steps show progress
    live. Pass quiet=True for the boot-polling loop, where refused connections
    are the expected case and printing each one buries the real output.
    """
    import subprocess

    cmd = ["ssh"] + _ssh_opts_argv(ssh_key, extra=[("ConnectTimeout", "10")])
    cmd += [f"{SSH_USER}@{ip}", command]
    result = subprocess.run(cmd, timeout=timeout, capture_output=quiet)
    return result.returncode


def wait_for_ssh(ip: str, timeout: int = SSH_WAIT_TIMEOUT,
                 ssh_key: str | None = None) -> bool:
    """Poll until sshd on the instance accepts a connection."""
    start = time.time()
    attempt = 0
    while time.time() - start < timeout:
        if instance_exec(ip, "true", timeout=15, ssh_key=ssh_key, quiet=True) == 0:
            return True
        attempt += 1
        # Report every ~60s so a slow boot looks like progress, not a hang.
        if attempt % 6 == 0:
            print(f"    still booting ({int(time.time() - start)}s elapsed)...", flush=True)
        time.sleep(10)
    return False


def install_git_key(ip: str, key_path: str, ssh_key: str | None = None) -> bool:
    """Install a private SSH key on the instance and bind it to github.com.

    Writes the key to ~/.ssh/id_github, adds a github.com block to ~/.ssh/config
    with IdentitiesOnly, and seeds known_hosts via ssh-keyscan so the clone
    never stops on an interactive host-verification prompt.
    """
    key_content = Path(key_path).read_text(encoding="utf-8")
    if not key_content.endswith("\n"):
        key_content += "\n"

    # The key travels inside a quoted heredoc ('GIT_KEY_EOF') so that $-signs
    # and backslashes in the key material are preserved literally.
    script = (
        "set -e\n"
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh\n"
        "cat > ~/.ssh/id_github <<'GIT_KEY_EOF'\n"
        f"{key_content}"
        "GIT_KEY_EOF\n"
        "chmod 600 ~/.ssh/id_github\n"
        "touch ~/.ssh/config && chmod 600 ~/.ssh/config\n"
        # Drop any github.com block from a previous run before appending ours,
        # so re-provisioning does not stack duplicate entries.
        "grep -q 'Host github.com' ~/.ssh/config || cat >> ~/.ssh/config <<'SSH_CFG_EOF'\n"
        "Host github.com\n"
        "    HostName github.com\n"
        "    User git\n"
        "    IdentityFile ~/.ssh/id_github\n"
        "    IdentitiesOnly yes\n"
        "SSH_CFG_EOF\n"
        "ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null || true\n"
        "chmod 600 ~/.ssh/known_hosts\n"
    )
    if instance_exec(ip, script, timeout=60, ssh_key=ssh_key) != 0:
        print("    FAILED to install git key", file=sys.stderr)
        return False
    print("    Installed git SSH key for github.com")
    return True


def create_user_workspaces(ip: str, dirs: list[str], repo_url: str = GIT_REPO_URL,
                           ssh_key: str | None = None) -> bool:
    """Create one directory per participant, each holding a clone of the repo.

    The repo is fetched from GitHub once into REPO_CACHE_DIR and copied into
    every participant directory. Both halves are idempotent: an existing cache
    is pulled (and rebuilt from scratch if a previous run left it without a
    valid HEAD), and a directory that already has a working clone is left alone,
    so re-running this never discards participant work.
    """
    import subprocess

    clone_dir = CLONE_DIR_NAME
    script = (
        "set -e\n"
        f"CACHE={REPO_CACHE_DIR}\n"
        # Phase 1: one network fetch, shared by every participant directory.
        'if [ -d "$CACHE/.git" ] && git -C "$CACHE" rev-parse HEAD >/dev/null 2>&1; then\n'
        '  echo "Repo cache present at $CACHE, pulling latest..."\n'
        '  git -C "$CACHE" pull --ff-only\n'
        "else\n"
        '  echo "No valid repo cache; cloning fresh..."\n'
        '  rm -rf "$CACHE"\n'
        f'  git clone --depth 1 {repo_url} "$CACHE"\n'
        "fi\n"
        # Phase 2: a local copy per participant. `cp -a` preserves .git, so each
        # copy is a fully functional working tree, not a shared checkout.
        f'for user in {" ".join(dirs)}; do\n'
        '  dest="$HOME/$user"\n'
        '  mkdir -p "$dest"\n'
        f'  if [ -d "$dest/{clone_dir}/.git" ]; then\n'
        f'    echo "  $user: {clone_dir} already present, leaving it alone"\n'
        "  else\n"
        f'    rm -rf "$dest/{clone_dir}"\n'
        f'    cp -a "$CACHE" "$dest/{clone_dir}"\n'
        f'    echo "  $user: created $dest/{clone_dir}"\n'
        "  fi\n"
        "done\n"
        f'echo "Workspaces ready: $(ls -d $HOME/{{{",".join(dirs)}}} 2>/dev/null | wc -l) directories"\n'
    )
    try:
        rc = instance_exec(ip, script, timeout=PROVISION_TIMEOUT, ssh_key=ssh_key)
    except subprocess.TimeoutExpired:
        print(f"    TIMED OUT after {PROVISION_TIMEOUT}s. The instance is fine; "
              f"re-run with --provision --instance <id> to finish.", file=sys.stderr)
        return False
    if rc != 0:
        print("    FAILED to create user workspaces", file=sys.stderr)
        return False
    print(f"    Created {len(dirs)} workspace(s), each with {clone_dir}/")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Provisioning and reporting
# ─────────────────────────────────────────────────────────────────────────────

def provision_instance(instance: dict, git_private_key: str | None,
                       dirs: list[str], ssh_key: str | None = None) -> bool:
    """Wait for SSH, install the git key, and lay out the participant workspaces."""
    ip = instance.get("ip")
    if not ip:
        print(f"    Instance {instance['id']} has no IP; cannot provision.",
              file=sys.stderr)
        return False

    if not git_private_key:
        print("  Note: --git-private-key not given, so no repo is cloned.",
              file=sys.stderr)
        return False

    print(f"\n  Waiting for sshd on {ip}...")
    if not wait_for_ssh(ip, ssh_key=ssh_key):
        print(f"    SSH never came up within {SSH_WAIT_TIMEOUT}s; skipping setup.",
              file=sys.stderr)
        return False
    print("    SSH ready!")

    if not install_git_key(ip, git_private_key, ssh_key=ssh_key):
        return False
    return create_user_workspaces(ip, dirs, ssh_key=ssh_key)


def print_connection_info(instance: dict, ssh_key: str | None = None) -> None:
    ssh_flags = _ssh_opts_flagstr(ssh_key or "<path-to-key>")
    print(f"\n  Instance: {instance.get('name') or '(unnamed)'} ({instance['id']})")
    print(f"  Type:     {(instance.get('instance_type') or {}).get('name', '?')}")
    print(f"  Region:   {(instance.get('region') or {}).get('name', '?')}")
    print(f"  Status:   {instance.get('status', 'unknown')}")
    if instance.get("ip"):
        print(f"  SSH:      ssh {SSH_USER}@{instance['ip']} {ssh_flags}")
    if instance.get("jupyter_url"):
        print(f"  Jupyter:  {instance['jupyter_url']}")


def print_instance_list(instances: list[dict], ssh_key: str | None = None) -> None:
    print(f"\n{'='*70}")
    print(f"  Bootcamp Instances ({len(instances)})")
    print(f"{'='*70}")
    for instance in sorted(instances, key=lambda i: i.get("name") or ""):
        print_connection_info(instance, ssh_key=ssh_key)


def bootcamp_instances() -> list[dict]:
    return [i for i in list_instances() if (i.get("name") or "").startswith(INSTANCE_NAME_PREFIX)]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Deploy a bootcamp instance on Lambda Cloud with per-user workspaces")
    parser.add_argument("--name", default=INSTANCE_NAME_PREFIX,
                        help=f"Instance name (default: {INSTANCE_NAME_PREFIX}). Must start "
                             f"with '{INSTANCE_NAME_PREFIX}' so --list/--terminate-all find it.")
    parser.add_argument("--instance-type", default=INSTANCE_TYPE,
                        help=f"Lambda instance type (default: {INSTANCE_TYPE})")
    parser.add_argument("--image-family", default=IMAGE_FAMILY,
                        help=f"Machine image family (default: {IMAGE_FAMILY})")
    parser.add_argument("--image-id", default=None,
                        help="Pin an exact machine image id, with or without "
                             "dashes. Overrides --image-family.")
    parser.add_argument("--region", default=None,
                        help="Region to launch in. Default: the first region "
                             "advertising capacity for the instance type.")
    parser.add_argument("--users", type=int, default=DEFAULT_USER_COUNT,
                        help=f"How many userN workspaces to create alongside 'test' "
                             f"(default: {DEFAULT_USER_COUNT})")
    parser.add_argument("--ssh-key", default=None,
                        help="Path to the SSH private key used for all instance SSH "
                             "operations. Its public key is registered with Lambda "
                             "(if not already present) and installed on the instance.")
    parser.add_argument("--git-private-key", default=None,
                        help=f"Path to an SSH private key installed on the instance and "
                             f"used to clone {GIT_REPO_URL}.")
    parser.add_argument("--list", action="store_true",
                        help="List bootcamp instances and exit.")
    parser.add_argument("--provision", action="store_true",
                        help="Provision existing instances instead of launching a new "
                             "one. Targets --instance, or every bootcamp instance. "
                             "Idempotent: the repo cache is pulled and existing "
                             "workspaces are left untouched.")
    parser.add_argument("--instance", action="append", default=[], dest="instances",
                        metavar="INSTANCE_ID",
                        help="Instance id to act on. Repeatable. Used with --provision.")
    parser.add_argument("--terminate", metavar="INSTANCE_ID",
                        help="Terminate one instance by id.")
    parser.add_argument("--terminate-all", action="store_true",
                        help="Terminate every bootcamp instance.")
    parser.add_argument("--api-key", default=None,
                        help="Lambda Cloud API key (or LAMBDA_API_KEY env / .env).")
    args = parser.parse_args()

    global API_KEY
    if args.api_key:
        API_KEY = args.api_key
    if not API_KEY:
        print("Error: set LAMBDA_API_KEY (env or .env) or pass --api-key", file=sys.stderr)
        sys.exit(1)

    for label, path in (("--ssh-key", args.ssh_key),
                        ("--git-private-key", args.git_private_key)):
        if path and not os.path.isfile(os.path.expanduser(path)):
            print(f"Error: {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    ssh_key = os.path.expanduser(args.ssh_key) if args.ssh_key else None
    git_private_key = (os.path.expanduser(args.git_private_key)
                       if args.git_private_key else None)
    dirs = participant_dirs(args.users)

    # ── List ──
    if args.list:
        instances = bootcamp_instances()
        if not instances:
            print("No bootcamp instances found.")
            return
        print_instance_list(instances, ssh_key=ssh_key)
        return

    # ── Terminate ──
    if args.terminate:
        terminate_instance(args.terminate)
        return

    if args.terminate_all:
        instances = bootcamp_instances()
        if not instances:
            print("No bootcamp instances to terminate.")
            return
        for instance in instances:
            terminate_instance(instance["id"])
        print(f"Terminated {len(instances)} instance(s).")
        return

    # ── Provision existing instances ──
    #
    # The resume path: the instance already exists (an earlier run was
    # interrupted, or sshd was not up in time), so skip the launch.
    if args.provision:
        if args.instances:
            targets = [get_instance(i) for i in args.instances]
        else:
            targets = bootcamp_instances()
            if not targets:
                print("No bootcamp instances found to provision.")
                return
        print(f"\n{'='*70}")
        print(f"  Provisioning {len(targets)} existing instance(s)")
        print(f"{'='*70}")
        failed = 0
        for instance in targets:
            print_connection_info(instance, ssh_key=ssh_key)
            if not provision_instance(instance, git_private_key, dirs, ssh_key=ssh_key):
                failed += 1
        sys.exit(1 if failed else 0)

    # ── Launch ──
    if not args.name.startswith(INSTANCE_NAME_PREFIX):
        print(f"Error: --name must start with '{INSTANCE_NAME_PREFIX}'", file=sys.stderr)
        sys.exit(1)
    if not ssh_key:
        print("Error: --ssh-key is required to launch (Lambda needs a key to "
              "inject, and we SSH in to provision).", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  Launching 1 instance '{args.name}'")
    print(f"  Type:  {args.instance_type}")
    print(f"  Image: {args.image_id or args.image_family}")
    print(f"  Users: {len(dirs)} workspaces ({dirs[0]}, {dirs[1]} ... {dirs[-1]})")
    print(f"{'='*70}\n")

    image = image_spec(args.image_id, args.image_family)
    region = pick_region(args.instance_type, args.region)
    ssh_key_name = ensure_ssh_key(derive_public_key(ssh_key))

    instance_id = launch_instance(args.name, args.instance_type, region,
                                  ssh_key_name, image)
    print(f"  Launched {instance_id} in {region}")

    instance = wait_for_instance(instance_id)
    print_connection_info(instance, ssh_key=ssh_key)

    if instance.get("status") != "active":
        print(f"\n  Instance is {instance.get('status')}, not active. Resume with:\n"
              f"    python {Path(__file__).name} --provision --instance {instance_id} "
              f"--ssh-key {args.ssh_key} --git-private-key {args.git_private_key}\n",
              file=sys.stderr)
        sys.exit(1)

    if not provision_instance(instance, git_private_key, dirs, ssh_key=ssh_key):
        # The instance exists and is billing; hand back the command that
        # finishes it rather than one that would launch a second box.
        print(f"\n{'='*70}")
        print("  Provisioning did not complete. Resume with:\n")
        print(f"    python {Path(__file__).name} --provision --instance {instance_id} "
              f"--ssh-key {args.ssh_key} --git-private-key {args.git_private_key}\n")
        print(f"{'='*70}", flush=True)
        sys.exit(1)

    ssh_flags = _ssh_opts_flagstr(args.ssh_key)
    print(f"""
{'='*70}
  PARTICIPANT INSTRUCTIONS
{'='*70}

  Connect over SSH:

    ssh {SSH_USER}@{instance['ip']} {ssh_flags}

  Each participant has their own workspace in the home directory, holding a
  full clone of the bootcamp repo:

    ~/test/{CLONE_DIR_NAME}     (scratch / instructor check)
    ~/user1/{CLONE_DIR_NAME}
    ...
    ~/{dirs[-1]}/{CLONE_DIR_NAME}

  Work in your assigned directory, e.g.:

    cd ~/user1/{CLONE_DIR_NAME}

  VS Code Remote-SSH: add the host with the ssh line above, connect, and open
  your own ~/userN/{CLONE_DIR_NAME} folder.
{'='*70}
""")


if __name__ == "__main__":
    main()
