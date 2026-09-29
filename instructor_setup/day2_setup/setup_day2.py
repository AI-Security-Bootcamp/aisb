#!/usr/bin/env python3
"""
Provision the Day 2 (AI control) lab environment on a fleet of VMs.

The fleet moves through one phase at a time, and every phase runs on all VMs in
parallel: each phase starts only once the previous one has finished everywhere,
so a failure is visible against the whole fleet rather than buried in one host's
log. A VM that fails a phase is dropped from the later ones and named in the
final summary.

    Setup (default)             Verification (--test-solutions)
    1. clone the repo           1. write the .env
    2. install requirements     2. run each Day 2 solution
    3. write the .env
    4. verify the environment

--test-solutions runs each section as a smoke test by default: a section that
gets through --smoke-seconds (60 by default) without erroring counts as a pass
and the run moves on to the next one. That catches the failures worth catching
on a fleet -- a broken import, a missing key, a bad install -- in a few minutes.
It does NOT catch a failure that happens later in a section, such as an
assertion on a result the section spends ten minutes computing. Pass --detail to
run every section to completion and judge it on its real exit code.

Usage:
    # Provision the fleet: clone, install requirements, write the .env, verify.
    python3 setup_day2.py --ssh-key ~/.ssh/aisb_key --git-private-key ~/.ssh/aisb_key

    # Smoke-test every Day 2 section on the fleet -- a few minutes, catches a
    # broken install, a missing key, or an import error.
    python3 setup_day2.py --test-solutions --ssh-key ~/.ssh/aisb_key \\
        --git-private-key ~/.ssh/aisb_key --openrouter-key sk-or-...

    # --detail instead runs every section to completion and judges it on its
    # real exit code. This is the one that catches a content bug -- an assertion
    # on a result the section spends minutes computing, such as
    # 2.2-monitoring's "monitor AUC > 0.5". Budget the best part of an hour per
    # VM: 2.3-control-protocols alone takes over half an hour, and every section
    # runs its evals against the API for real. Use it before a cohort, and after
    # changing a Day 2 solution file -- not as the routine fleet check.
    python3 setup_day2.py --test-solutions --detail --ssh-key ~/.ssh/aisb_key \\
        --git-private-key ~/.ssh/aisb_key --openrouter-key sk-or-...

    # One VM only, with a longer smoke window.
    python3 setup_day2.py --test-solutions --smoke-seconds 180 \\
        --ssh-key ~/.ssh/aisb_key -f one_host.txt --openrouter-key sk-or-...

As in deploy_runpod.py, the two key roles are separate flags: --ssh-key reaches
the VMs, --git-private-key is installed on each VM to clone the private repo.
Passing the same key for both is the simplest setup.

Day 2 is API-only, so every exercise needs an OpenRouter key. --openrouter-key
supplies it, falling back to the OPENROUTER_API_KEY environment variable and
then to OPENROUTER_API_KEY in the repo-root .env. Unlike deploy_runpod.py's
pods, the .env is written to stay: participants run the exercises on these VMs.
Use --no-env to leave it alone, or --env-file to upload a complete .env verbatim.

instance.txt holds one instance per line, "[user@]host[:port]". Blank lines and
lines starting with # are ignored.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Our prints go through Python's stdout while the remote commands we drive (apt,
# git, pip) write to the inherited fd directly. Line buffering keeps the two
# streams in order when the run is piped to tee or a log file.
sys.stdout.reconfigure(line_buffering=True)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# This script lives at <repo>/instructor_setup/day2_setup/setup_day2.py
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_INSTANCE_FILE = SCRIPT_DIR / "instance.txt"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

# The bootcamp repo is private, so the default clone URL is the SSH form and the
# VM clones with the deploy key installed by git_key_script(). Pass --repo-url to
# override (an https URL only works for a VM with cached credentials).
GIT_REPO_URL = "git@github.com:AI-Security-Bootcamp/aisb.git"
DEFAULT_BRANCH = "main"
DEFAULT_DEST = "aisb"  # relative to the remote $HOME

# Day 2 gets its own virtualenv: control-arena needs datasets>=4.4.2, which
# conflicts with the datasets==3.6.0 pin in the root requirements.txt. Day 2 is
# API-only, so this env deliberately skips torch.
VENV_DIR = ".venv-day2"
REQUIREMENTS = [
    "requirements-control-arena.txt",
    "2.1-coding-agents/requirements.txt",
]

# 2.1 and 2.4 are discussion sections and define no tests.
DAY2_TEST_FILES = [
    "2.2-monitoring/section2_test.py",
    "2.3-control-protocols/section3_test.py",
]

# aisb_utils/test_solutions.py rewrites each solution file with libcst and
# formats it with black; neither is in the Day 2 requirements, because only this
# instructor-side check needs them.
TEST_SOLUTIONS_BUILD_TOOLING = "libcst~=1.8.0 black~=25.1.0 termcolor~=3.1.0"

SSH_BASE_OPTS = [
    ("BatchMode", "yes"),
    ("StrictHostKeyChecking", "accept-new"),
    ("ConnectTimeout", "15"),
    # Keep ssh-agent from offering other keys ahead of the one we were given.
    ("IdentitiesOnly", "yes"),
]

CLONE_TIMEOUT = 600         # apt update/install on a cold VM, then the clone
REQUIREMENTS_TIMEOUT = 1800  # control-arena's dependency tree is large
VERIFY_TIMEOUT = 600
# Two nested budgets for --test-solutions --detail. The section timeout is what
# test_solutions.py's own watchdog uses per section; its 1800s default kills
# 2.3-control-protocols partway through its threshold sweeps, so raise it here.
# The run timeout covers all four Day 2 sections on one VM.
DEFAULT_SECTION_TIMEOUT = 3600
DEFAULT_TEST_TIMEOUT = 9000

# How long a section gets to prove itself in the default (smoke) mode.
DEFAULT_SMOKE_SECONDS = 60


# ─────────────────────────────────────────────────────────────────────────────
# Instances
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Instance:
    """One SSH target from instance.txt."""

    target: str        # [user@]host
    port: int | None   # optional :port suffix

    def __str__(self) -> str:
        return self.target if self.port is None else f"{self.target}:{self.port}"


def parse_instances(path: Path) -> list[Instance]:
    """Read "[user@]host[:port]" lines, ignoring blanks and # comments."""
    instances: list[Instance] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        target, port = line, None
        # Only split a trailing :NNNN -- an IPv6 literal is full of colons.
        head, sep, tail = line.rpartition(":")
        if sep and tail.isdigit():
            target, port = head, int(tail)
        if not target:
            sys.exit(f"{path}:{lineno}: cannot parse instance {raw!r}")
        instances.append(Instance(target, port))
    return instances


# ─────────────────────────────────────────────────────────────────────────────
# SSH plumbing
# ─────────────────────────────────────────────────────────────────────────────

def ssh_cmd(inst: Instance, ssh_key: str) -> list[str]:
    cmd = ["ssh", "-i", ssh_key]
    for key, value in SSH_BASE_OPTS:
        cmd += ["-o", f"{key}={value}"]
    if inst.port is not None:
        cmd += ["-p", str(inst.port)]
    return cmd + [inst.target]


def scp_cmd(inst: Instance, ssh_key: str) -> list[str]:
    cmd = ["scp", "-q", "-i", ssh_key]
    for key, value in SSH_BASE_OPTS:
        cmd += ["-o", f"{key}={value}"]
    if inst.port is not None:
        # scp spells the port -P, not -p (-p preserves timestamps).
        cmd += ["-P", str(inst.port)]
    return cmd


def run(argv: list[str], stdin: str | None, timeout: int) -> tuple[int, str]:
    """Run a command, capturing its output.

    Phases run concurrently, so output is always captured and printed as one
    block per host afterwards; interleaving it live would make it impossible to
    tell which VM a traceback came from.
    """
    try:
        result = subprocess.run(
            argv,
            # ssh forwards its own stdin to the remote command, so a call with
            # nothing to send (`ssh host chmod ...`, scp) must be given
            # /dev/null explicitly -- otherwise it inherits ours and blocks,
            # swallowing the operator's terminal input.
            input=stdin,
            stdin=None if stdin is not None else subprocess.DEVNULL,
            text=True,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.TimeoutExpired as e:
        partial = e.output or ""
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        return 124, partial + f"\nTIMEOUT after {timeout}s"
    except FileNotFoundError as e:
        return 127, str(e)
    return result.returncode, result.stdout or ""


def ssh_script(inst: Instance, ssh_key: str, script: str, timeout: int) -> tuple[int, str]:
    """Execute a bash script on the VM by piping it to `bash -s` over SSH."""
    return run(ssh_cmd(inst, ssh_key) + ["bash -s"], script, timeout)


# ─────────────────────────────────────────────────────────────────────────────
# Remote scripts
# ─────────────────────────────────────────────────────────────────────────────

def git_key_script(key_path: Path) -> str:
    """Install a private key on the VM and bind it to github.com.

    The key is passed through a quoted heredoc so $-expansion and backslashes in
    the key material stay literal.
    """
    key = key_path.read_text()
    if not key.endswith("\n"):
        key += "\n"
    return (
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh\n"
        "cat > ~/.ssh/id_github <<'GIT_KEY_EOF'\n"
        f"{key}"
        "GIT_KEY_EOF\n"
        "chmod 600 ~/.ssh/id_github\n"
        "touch ~/.ssh/config && chmod 600 ~/.ssh/config\n"
        # Append only once, so re-running does not stack duplicate blocks.
        "grep -q 'IdentityFile ~/.ssh/id_github' ~/.ssh/config || cat >> ~/.ssh/config <<'SSH_CFG_EOF'\n"
        "Host github.com\n"
        "    HostName github.com\n"
        "    User git\n"
        "    IdentityFile ~/.ssh/id_github\n"
        "    IdentitiesOnly yes\n"
        "SSH_CFG_EOF\n"
        "ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null || true\n"
        "chmod 600 ~/.ssh/known_hosts\n"
        "echo '  Installed the GitHub deploy key.'\n"
    )


def clone_script(repo_url: str, branch: str, dest: str, git_key: Path | None) -> str:
    """Phase 1: base packages, deploy key, and the repo itself."""
    url, br, ds = shlex.quote(repo_url), shlex.quote(branch), shlex.quote(dest)
    return f"""set -euo pipefail
REPO_URL={url}; BRANCH={br}; DEST="$HOME"/{ds}

echo "--- Base packages (git, python3-venv, pip) ---"
missing=()
command -v git >/dev/null 2>&1 || missing+=(git)
python3 -c 'import venv' >/dev/null 2>&1 || missing+=(python3-venv)
python3 -m pip --version >/dev/null 2>&1 || missing+=(python3-pip)
if [ "${{#missing[@]}}" -gt 0 ]; then
    SUDO=""
    if [ "$(id -u)" -ne 0 ]; then
        sudo -n true >/dev/null 2>&1 || {{ echo "FATAL: need ${{missing[*]}} but have no passwordless sudo." >&2; exit 1; }}
        SUDO="sudo -n"
    fi
    $SUDO apt-get update -qq
    $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${{missing[@]}}"
else
    echo "  Already present."
fi

{git_key_script(git_key) if git_key else ""}
echo "--- Repo: $REPO_URL ($BRANCH) -> $DEST ---"
if [ -d "$DEST/.git" ]; then
    git -C "$DEST" fetch --quiet origin "$BRANCH"
    git -C "$DEST" checkout --quiet "$BRANCH"
    # Fast-forward only: never discard work a participant already has on the VM.
    git -C "$DEST" merge --ff-only "origin/$BRANCH" \\
        || echo "  WARNING: cannot fast-forward $DEST -- leaving the checkout as it is."
else
    git clone --quiet --branch "$BRANCH" "$REPO_URL" "$DEST"
fi
git -C "$DEST" --no-pager log -1 --oneline
"""


def requirements_script(dest: str) -> str:
    """Phase 2: build the Day 2 virtualenv and install its dependencies."""
    installs = "\n".join(
        f'"$VENV"/bin/python -m pip install --quiet -r {shlex.quote(req)}' for req in REQUIREMENTS
    )
    return f"""set -euo pipefail
DEST="$HOME"/{shlex.quote(dest)}
VENV="$DEST"/{shlex.quote(VENV_DIR)}

echo "--- Day 2 virtualenv ($VENV) ---"
cd "$DEST"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV"/bin/python -m pip install --quiet --upgrade pip
{installs}
echo "  Installed: {' '.join(REQUIREMENTS)}"
"""


def env_script(dest: str, openrouter_key: str) -> str:
    """Write the repo-root .env holding the OpenRouter key.

    aisb_utils/env.py's load_dotenv() requires that file, so every Day 2
    exercise depends on it. The key travels inside the script we pipe to the
    VM's stdin over SSH, so it never appears in a command line (and therefore
    never in the VM's process list or the operator's shell history), and the
    quoted heredoc delimiter keeps the shell from expanding it.
    """
    return f"""set -euo pipefail
DEST="$HOME"/{shlex.quote(dest)}
cat > "$DEST"/.env <<'AISB_ENV_EOF'
OPENROUTER_API_KEY={openrouter_key}
AISB_ENV_EOF
chmod 600 "$DEST"/.env
echo "  Wrote $DEST/.env"
"""


# Runs on the VM inside the Day 2 venv. Kept as a plain string (not an f-string)
# so its own braces need no escaping.
_ENV_CHECK_PY = '''
import importlib.metadata as md
import os
import sys

for pkg in ["control-arena", "inspect-ai", "numpy", "matplotlib", "pytest"]:
    try:
        print(f"  {pkg}: {md.version(pkg)}")
    except md.PackageNotFoundError:
        sys.exit(f"  FATAL: {pkg} is not installed in the Day 2 venv.")

sys.path.insert(0, os.getcwd())
try:
    from aisb_utils.env import load_dotenv
    load_dotenv()
except FileNotFoundError as e:
    print(f"  WARNING: {e}")
else:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    print(f"  OPENROUTER_API_KEY: {key[:6] + '...' if key else 'MISSING'}")
'''


def verify_script(dest: str) -> str:
    """Phase 4: package versions, .env, and Day 2 test collection."""
    files = " ".join(shlex.quote(f) for f in DAY2_TEST_FILES)
    return f"""set -euo pipefail
cd "$HOME"/{shlex.quote(dest)}
PY={shlex.quote(f"./{VENV_DIR}/bin/python")}

echo "--- Environment check ---"
"$PY" - <<'ENV_CHECK_EOF'
{_ENV_CHECK_PY}
ENV_CHECK_EOF

echo "--- Collecting the Day 2 exercise tests ---"
# The test functions take the participant's solution as an argument, so they are
# not runnable unattended. Collection still exercises every heavy import
# (control_arena, inspect_ai, day2_utils, matplotlib), which is what a bad
# install actually breaks. --test-solutions runs the solutions for real.
"$PY" -m pytest --collect-only -q {files}
echo "  2.1-coding-agents and 2.4-safety-simulator are discussion sections with no tests."
"""


# Generates one section's answers file on the VM, run with (solution, answers,
# test-module) as arguments. Kept as a plain string so its braces need no
# escaping; the same build_reference_py that test_solutions.py uses.
_GENERATE_ANSWERS_PY = """
import sys
sys.path.insert(0, ".")
from aisb_utils.solution_parsing import build_reference_py

solution, answers, test_module = sys.argv[1:4]
with open(solution) as infile, open(answers, "w") as outfile:
    build_reference_py(infile, outfile, test_module)
"""


def smoke_script(dest: str, smoke_seconds: int) -> str:
    """Give each Day 2 section `smoke_seconds` to fail, and pass it if it doesn't.

    Same preparation as test_solutions.py -- fill in every `if "SOLUTION":`
    branch, run the section from its own folder, delete the answers file -- but
    a section still running when the timer expires is reported as a pass rather
    than killed and failed. Enough to catch a broken install, a missing key, or
    an import error across a fleet in a few minutes; blind to anything that
    fails later in a section (use --detail for that).
    """
    return f"""set -uo pipefail
cd "$HOME"/{shlex.quote(dest)}
PY="$HOME"/{shlex.quote(dest)}/{shlex.quote(f"{VENV_DIR}/bin/python")}
SMOKE={smoke_seconds}

# The answers files hold every solution, so sweep them whatever happens -- a
# failed section, a killed ssh, or the timeout below.
trap 'rm -f "$HOME"/{shlex.quote(dest)}/2.*/*_answers.py' EXIT

echo "--- Build tooling for the answers build ---"
"$PY" -m pip install --quiet {TEST_SOLUTIONS_BUILD_TOOLING}

failed=0
for solution in 2.*/section*_solution.py; do
    section=$(dirname "$solution")
    stem=$(basename "$solution" _solution.py)
    answers="$section/${{stem}}_answers.py"

    echo ""
    echo "--- $section (up to ${{SMOKE}}s) ---"
    "$PY" - "$solution" "$answers" "${{stem}}_test.py" <<'GEN_ANSWERS_EOF'
{_GENERATE_ANSWERS_PY}
GEN_ANSWERS_EOF

    # Run from the section folder, exactly as a participant would, so the
    # answers file's `from sectionN_test import ...` resolves.
    output=$(cd "$section" && timeout -k 5 "$SMOKE" "$PY" "${{stem}}_answers.py" 2>&1)
    rc=$?
    rm -f "$answers"

    if [ "$rc" -eq 0 ]; then
        echo "  PASS -- ran to completion"
    # 124 is a plain timeout; 137 is the -k SIGKILL for a section that ignored
    # the TERM. Both mean "still running when the timer expired", not a failure.
    elif [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
        echo "  PASS -- no error in ${{SMOKE}}s (still running, not waited out)"
    else
        echo "  FAIL -- exit $rc"
        echo "$output" | tail -25 | sed 's/^/    /'
        failed=$((failed + 1))
    fi
done

echo ""
if [ "$failed" -gt 0 ]; then
    echo "$failed Day 2 section(s) failed."
    exit 1
fi
echo "Every Day 2 section passed the smoke test."
"""


def test_solutions_script(dest: str, section_timeout: int) -> str:
    """Run every Day 2 reference solution end-to-end on the VM.

    aisb_utils/test_solutions.py fills in each `if "SOLUTION":` branch, runs the
    section against the VM's real environment, and deletes the generated
    answers file whether it passed or failed.
    """
    return f"""set -euo pipefail
cd "$HOME"/{shlex.quote(dest)}
PY={shlex.quote(f"./{VENV_DIR}/bin/python")}

echo "--- Build tooling for test_solutions.py ---"
"$PY" -m pip install --quiet {TEST_SOLUTIONS_BUILD_TOOLING}

echo "--- Running every Day 2 solution ---"
"$PY" aisb_utils/test_solutions.py --day 2 --timeout {section_timeout}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Phases
# ─────────────────────────────────────────────────────────────────────────────

# A phase is one step run against one VM; the driver runs it across the fleet.
PhaseFn = Callable[[Instance, argparse.Namespace], tuple[int, str]]


@dataclass
class Phase:
    name: str
    fn: PhaseFn


def phase_clone(inst: Instance, args: argparse.Namespace) -> tuple[int, str]:
    git_key = Path(args.git_private_key) if args.git_private_key else None
    script = clone_script(args.repo_url, args.branch, args.dest, git_key)
    return ssh_script(inst, args.ssh_key, script, CLONE_TIMEOUT)


def phase_requirements(inst: Instance, args: argparse.Namespace) -> tuple[int, str]:
    return ssh_script(inst, args.ssh_key, requirements_script(args.dest), REQUIREMENTS_TIMEOUT)


def phase_env(inst: Instance, args: argparse.Namespace) -> tuple[int, str]:
    if args.upload_env_file:
        remote = f"{inst.target}:{shlex.quote(args.dest)}/.env"
        rc, out = run(scp_cmd(inst, args.ssh_key) + [args.upload_env_file, remote], None, 60)
        if rc != 0:
            return rc, out
        chmod = f"chmod 600 {shlex.quote(args.dest)}/.env"
        rc, chmod_out = run(ssh_cmd(inst, args.ssh_key) + [chmod], None, 30)
        return rc, out + chmod_out + f"  Uploaded {args.upload_env_file}\n"
    return ssh_script(inst, args.ssh_key, env_script(args.dest, args.openrouter_key), 30)


def phase_verify(inst: Instance, args: argparse.Namespace) -> tuple[int, str]:
    return ssh_script(inst, args.ssh_key, verify_script(args.dest), VERIFY_TIMEOUT)


def phase_test_solutions(inst: Instance, args: argparse.Namespace) -> tuple[int, str]:
    if args.detail:
        script = test_solutions_script(args.dest, args.section_timeout)
        return ssh_script(inst, args.ssh_key, script, args.timeout)
    script = smoke_script(args.dest, args.smoke_seconds)
    # Four sections, each capped at smoke_seconds, plus the answers build and a
    # generous margin for the pip check and SSH itself.
    return ssh_script(inst, args.ssh_key, script, 4 * args.smoke_seconds + 600)


def build_phases(args: argparse.Namespace) -> list[Phase]:
    """The phase list for this run, in order."""
    writes_env = bool(args.openrouter_key or args.upload_env_file)
    if args.test_solutions:
        phases = [Phase("Write .env", phase_env)] if writes_env else []
        label = ("Test solutions (full run)" if args.detail
                 else f"Smoke-test solutions ({args.smoke_seconds}s per section)")
        return phases + [Phase(label, phase_test_solutions)]

    phases = [
        Phase("Clone the repo", phase_clone),
        Phase("Install requirements", phase_requirements),
    ]
    if writes_env:
        phases.append(Phase("Write .env", phase_env))
    if not args.skip_verify:
        phases.append(Phase("Verify the environment", phase_verify))
    return phases


def run_phase(phase: Phase, instances: list[Instance], args: argparse.Namespace,
              index: int, total: int) -> list[Instance]:
    """Run one phase across the fleet in parallel; return the VMs that passed.

    Each host's output is printed as one block once that host finishes, so a
    traceback is never interleaved with another VM's.
    """
    jobs = args.jobs or len(instances)
    print(f"\n{'=' * 70}")
    print(f"  Phase {index}/{total}: {phase.name} -- {len(instances)} VM(s), "
          f"{min(jobs, len(instances))} at a time")
    print("=" * 70)

    passed: list[Instance] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(phase.fn, inst, args): inst for inst in instances}
        for future in concurrent.futures.as_completed(futures):
            inst = futures[future]
            rc, out = future.result()
            print(f"\n  --- {inst}")
            for line in out.splitlines():
                print(f"    {line}")
            if rc == 0:
                print(f"  --- {inst}: OK")
                passed.append(inst)
            else:
                print(f"  --- {inst}: FAILED (exit {rc})", file=sys.stderr)

    failed = len(instances) - len(passed)
    print(f"\n  Phase {index}/{total} ({phase.name}): {len(passed)}/{len(instances)} OK"
          + (f", {failed} failed -- dropped from the remaining phases" if failed else ""))
    return sorted(passed, key=str)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def read_openrouter_key(env_file: Path) -> str | None:
    """Read OPENROUTER_API_KEY out of a .env file, if it holds one.

    Uses python-dotenv when it is installed (the same parser aisb_utils/env.py
    relies on) and falls back to a minimal parse, so provisioning works from a
    laptop that has not installed the bootcamp requirements.
    """
    if not env_file.is_file():
        return None
    try:
        from dotenv import dotenv_values
        return dotenv_values(env_file).get("OPENROUTER_API_KEY") or None
    except ImportError:
        pass
    for line in env_file.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        name, sep, value = line.partition("=")
        if sep and name.strip() == "OPENROUTER_API_KEY":
            return value.strip().strip("\"'") or None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Provision the Day 2 lab environment on the VMs listed in instance.txt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-i", "--ssh-key", required=True,
                        help="path to an SSH private key used for all VM SSH operations")
    parser.add_argument("-f", "--instances", default=str(DEFAULT_INSTANCE_FILE),
                        help=f"instance list (default: {DEFAULT_INSTANCE_FILE})")
    parser.add_argument("--git-private-key",
                        help="path to an SSH private key. Installed on each VM and used to "
                             f"clone {GIT_REPO_URL}. Often the same key as --ssh-key.")
    parser.add_argument("--test-solutions", action="store_true",
                        help="skip provisioning; run every Day 2 reference solution on each VM "
                             "and report failures. Smoke test by default: a section that "
                             "survives --smoke-seconds counts as a pass")
    parser.add_argument("--detail", action="store_true",
                        help="with --test-solutions, run every section to completion "
                             "(aisb_utils/test_solutions.py --day 2) instead of smoke-testing "
                             "it. Slow -- 2.3-control-protocols alone takes over half an hour")
    parser.add_argument("--smoke-seconds", type=int, default=DEFAULT_SMOKE_SECONDS,
                        help="how long a section gets to fail before it counts as a pass "
                             f"(default: {DEFAULT_SMOKE_SECONDS}; ignored with --detail)")
    parser.add_argument("--openrouter-key",
                        help="OpenRouter API key for the Day 2 exercises. Written to the "
                             "repo-root .env on each VM. Falls back to the OPENROUTER_API_KEY "
                             f"environment variable, then to {DEFAULT_ENV_FILE}.")
    parser.add_argument("--env-file",
                        help="upload this complete .env file verbatim instead of writing one "
                             "that holds only the OpenRouter key")
    parser.add_argument("--no-env", action="store_true",
                        help="leave the VMs' .env alone")
    parser.add_argument("--repo-url", default=GIT_REPO_URL,
                        help=f"repo to clone (default: {GIT_REPO_URL})")
    parser.add_argument("--branch", default=DEFAULT_BRANCH,
                        help=f"branch to check out (default: {DEFAULT_BRANCH})")
    parser.add_argument("--dest", default=DEFAULT_DEST,
                        help=f"clone directory on the VM, relative to $HOME (default: {DEFAULT_DEST})")
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip the post-install environment check")
    parser.add_argument("--section-timeout", type=int, default=DEFAULT_SECTION_TIMEOUT,
                        help="with --detail, the per-section timeout handed to "
                             f"test_solutions.py (default: {DEFAULT_SECTION_TIMEOUT}; its own "
                             "default of 1800 is too short for 2.3-control-protocols)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TEST_TIMEOUT,
                        help=f"with --detail, the timeout for the whole run on one VM, in "
                             f"seconds (default: {DEFAULT_TEST_TIMEOUT})")
    parser.add_argument("-j", "--jobs", type=int, default=0,
                        help="cap how many VMs a phase touches at once (default: all of them)")
    args = parser.parse_args()

    if not Path(args.ssh_key).is_file():
        return fail(f"private key not found: {args.ssh_key}")
    if args.git_private_key and not Path(args.git_private_key).is_file():
        return fail(f"--git-private-key file not found: {args.git_private_key}")
    if not args.git_private_key and not args.test_solutions:
        # The repo is private and the clone runs on the VM, so without a deploy
        # key the clone only works on a VM that already has GitHub access.
        print("Warning: no --git-private-key given; the clone will fail unless the VMs\n"
              "         already have GitHub access or the repo is already cloned.",
              file=sys.stderr)
    instance_file = Path(args.instances)
    if not instance_file.is_file():
        return fail(f"instance file not found: {instance_file}")

    # Where the OpenRouter key comes from, in deploy_runpod.py's order: the flag,
    # then the environment, then the repo-root .env.
    args.upload_env_file = None
    args.key_source = None
    if args.no_env:
        args.openrouter_key = None
    elif args.env_file:
        if not Path(args.env_file).is_file():
            return fail(f"--env-file not found: {args.env_file}")
        args.upload_env_file = args.env_file
        args.key_source = args.env_file
    else:
        if args.openrouter_key:
            args.key_source = "--openrouter-key"
        elif os.environ.get("OPENROUTER_API_KEY"):
            args.openrouter_key = os.environ["OPENROUTER_API_KEY"]
            args.key_source = "the OPENROUTER_API_KEY environment variable"
        else:
            args.openrouter_key = read_openrouter_key(DEFAULT_ENV_FILE)
            args.key_source = str(DEFAULT_ENV_FILE)
        if not args.openrouter_key:
            args.key_source = None
            # Every Day 2 solution calls the API, so a test run without a key
            # only produces noise. Provisioning can still proceed.
            if args.test_solutions:
                return fail("no OpenRouter key found. Pass --openrouter-key, or --no-env if "
                            "the VMs already have a working .env.")
            print("Warning: no OpenRouter key found -- the Day 2 exercises will not run on the\n"
                  "         VMs. Pass --openrouter-key, or --no-env to silence this.",
                  file=sys.stderr)

    instances = parse_instances(instance_file)
    if not instances:
        return fail(f"no instances found in {instance_file}")

    phases = build_phases(args)
    mode = "Testing Day 2 solutions on" if args.test_solutions else "Day 2 setup:"
    print(f"{mode} {len(instances)} VM(s) from {instance_file}")
    if args.key_source:
        print(f"OpenRouter key from {args.key_source}")

    remaining = instances
    for index, phase in enumerate(phases, start=1):
        remaining = run_phase(phase, remaining, args, index, len(phases))
        if not remaining:
            print("\nEvery VM failed; stopping.", file=sys.stderr)
            break

    ok = {str(inst) for inst in remaining}
    failed = [str(inst) for inst in instances if str(inst) not in ok]
    print(f"\n{'=' * 70}")
    print(f"  Summary: {len(ok)}/{len(instances)} VM(s) completed every phase")
    for name in failed:
        print(f"    FAILED: {name}")
    print("=" * 70)
    if failed:
        return 1
    if not args.test_solutions:
        print("\nParticipants activate the environment with:")
        print(f"    cd ~/{args.dest} && source {VENV_DIR}/bin/activate")
        print("\nVerify the solutions actually run with:")
        print(f"    python3 {Path(__file__).name} --test-solutions "
              f"--ssh-key {args.ssh_key} --openrouter-key <key>")
    return 0


def fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
