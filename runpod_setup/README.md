# Bootcamp pod provisioning

Instructor workflow for provisioning RunPod pods for the bootcamp.

| File | Lives in | Runs on | Purpose |
| --- | --- | --- | --- |
| `deploy_runpod.py` | `runpod_setup/` | your laptop | Create, provision, list, test, and terminate pods |
| `setup_pod.sh` | `3.1-tokenization/setup/` | the pod | Full (GPU-day) setup: pinned deps, GPU check, model pre-download |
| `download_models.py` | `3.1-tokenization/setup/` | the pod | Fetch every tokenizer and model the exercises load |
| `run_exercise.sh` | `3.1-tokenization/setup/` | the pod, by participants | `setup` re-runs provisioning after a failure |
| `install_vscode_extensions.sh` | `3.1-tokenization/setup/` | the pod | Pre-install the VS Code *server* extensions (Python, Pylance, Jupyter) so Remote-SSH works on first connect |

See [3.1-tokenization/setup/README.md](../3.1-tokenization/setup/README.md) for how the
pod-side scripts fit together and which invariants they depend on.

The pod-side scripts are committed, because the pods reach them through their clone
at `/workspace/aisb/3.1-tokenization/setup`. `deploy_runpod.py` sits in the repo-root
`runpod_setup/` folder, which is gitignored, so a fresh clone will not contain it —
get it from another instructor before provisioning. Credentials stay out of the repo
through the SSH-key and `.env` rules in `.gitignore`.

Nothing is uploaded to the pods over SSH: they get every script from the clone. An
edit to `setup_pod.sh`, `download_models.py`, `requirements*.txt`, or any solution
file must be **pushed** before the next deploy, provision, or `--test-solutions` run
picks it up (the clone step pulls on a pod that already has the repo).

## Prerequisites

- `RUNPOD_API_KEY` set in the repo-root `.env` file (or exported in the shell).
- An SSH key pair used for participant access *and* for cloning the repo onto each pod.
- `OPENROUTER_API_KEY` (in `.env` or passed with `--openrouter-key`) if you want to
  run `--test-solutions` for the API-only days.

The bootcamp repo is **private**, so each pod clones it over SSH and needs a key:
`--git-private-key` installs one at `~/.ssh/id_github` on the pod and binds it to
`github.com`. Register its public half as a GitHub **deploy key** (repo → *Settings* →
*Deploy keys*, *Allow write access* unchecked).

The examples below use one key for both roles (`--ssh-key` for your own access,
`--git-private-key` for the clone), which is the simplest setup. Note the tradeoff:
participants have root on their pods, so assume they can read the deploy key —
keep it read-only and revoke it from GitHub once the session ends. Use two separate
keys if that matters for a given cohort.

### Generating the key

Use `-N ""` (no passphrase) — this script drives the pods non-interactively, so a
passphrase would hang provisioning.

```bash
ssh-keygen -t ed25519 -C "aisb-participant" -f ~/.ssh/aisb_key -N ""
```

Then add `~/.ssh/aisb_key.pub` to the repo's deploy keys. `--ssh-key` injects the
public half into every pod at creation time as `PUBLIC_KEY`.

## 1. Create pods in bulk

`--count` creates N pods, all sharing the same injected public key. Each pod then
clones the repo into `/workspace/aisb` and is set up according to `--day`.

**Days 1 and 2 (API-only):**

```bash
python runpod_setup/deploy_runpod.py --count 12 --day 1 \
    --ssh-key ~/.ssh/aisb_key --git-private-key ~/.ssh/aisb_key
```

**Days 3+ (GPU):** omit `--day` to get the full `setup_pod.sh` profile.

```bash
python runpod_setup/deploy_runpod.py --count 12 \
    --ssh-key ~/.ssh/aisb_key --git-private-key ~/.ssh/aisb_key
```

`--ssh-key` is what lets the script reach the pod at all; without it, pods are
created but arrive empty. `--git-private-key` is what lets the pod clone the repo;
without it, pods are created and listed but never cloned or set up.

### Setup profiles (`--day`)

`--day` at creation/provision time selects how much gets installed:

| Request | Profile | Installs | Time |
| --- | --- | --- | --- |
| `--day 1` | lightweight | `1.1-llm-internals/requirements.txt` + notebook tooling | ~1 min/pod |
| `--day 2` | lightweight | `requirements-control-arena.txt` + notebook tooling | ~1 min/pod |
| no `--day`, `--all`, `--day 3`+, or several days | full `setup_pod.sh` | Day 3 pinned deps, GPU check, model cache | several min/pod |

The lightweight profile skips the GPU check and the model downloads, and invokes
`install_vscode_extensions.sh` itself (on the full path `setup_pod.sh` does that).

**Days 1 and 2 must not share a pod.** control-arena's dependency stack conflicts
with the `datasets` pin the Day 1 / Day 3 requirements use, which is why a mixed
request (`--day 1 --day 2`) deliberately falls through to full setup instead of
trying to install both. Deploy a separate fleet, or re-provision between days.

### Resuming an interrupted deploy

Pod creation and provisioning are separate phases: `--count N` creates all N pods
first, then clones onto each (phase 1), then runs setup on each (phase 2). If a later
phase is interrupted — a transient RunPod API error, a Ctrl-C, a pod whose sshd is not
up in time — the pods still exist. **Do not re-run the plain deploy command**; that
creates a second fleet. Provision the existing pods instead:

```bash
# Finish every bootcamp pod that exists (idempotent: clones pull, setup re-runs)
python runpod_setup/deploy_runpod.py --provision --day 1 \
    --ssh-key ~/.ssh/aisb_key --git-private-key ~/.ssh/aisb_key

# Or just the ones that were skipped
python runpod_setup/deploy_runpod.py --provision --day 1 \
    --pod <pod_id> --pod <pod_id> \
    --ssh-key ~/.ssh/aisb_key --git-private-key ~/.ssh/aisb_key
```

`--provision` honors the same `--day` profiles, so it is also how you re-point an
existing fleet at a different day. When a run leaves pods unprovisioned it prints the
exact `--provision` command to resume them.

### Host maintenance

RunPod schedules maintenance on *physical hosts*, and the console then shows a
"down for maintenance" banner on every pod running there — the pod itself is
healthy until the window opens, at which point the host goes down and takes the
pod with it.

`--list` flags any affected pod:

```
  *** MAINTENANCE: host down 2026-08-05T14:00:00.000Z -> 2026-08-06T18:00:00.000Z
```

Pod creation checks this automatically: a pod that lands on a flagged host is
terminated and creation retried on a different GPU type. This matters because
RunPod schedules onto whichever machine has free capacity, and a machine being
drained for maintenance has plenty — so a naive retry lands straight back on the
host you are trying to escape. Different GPU models live on different machines,
so changing the requested type is what actually moves you. Pass
`--allow-maintenance-host` to accept a flagged pod anyway (it is created with a
warning).

New pods continue the fleet's numbering: with `aisb-bootcamp-01` through `-10`
running, `--count 2` creates `-11` and `-12`. Numbers freed by a terminated pod
are *not* reused, so a stale handout line can never point at someone else's pod.
Pass `--name` to override the numbering (for instance to reuse the name of a pod
you just replaced).

To replace a pod already caught in a window, terminate it and create a
replacement under the same name:

```bash
python runpod_setup/deploy_runpod.py --stop <pod_id>
python runpod_setup/deploy_runpod.py --count 1 --name aisb-bootcamp-<NN> --day 1 \
    --ssh-key ~/.ssh/aisb_key --git-private-key ~/.ssh/aisb_key
```

If a pod's SSH never comes up, check its `--list` entry for an `SSH:` line. A pod
with **no** `SSH:` line was never given a public port-22 mapping, so no amount of
waiting will help — terminate it and create a replacement the same way.

List pods (with SSH/Jupyter connection strings) at any time:

```bash
python runpod_setup/deploy_runpod.py --list --ssh-key ~/.ssh/aisb_key
```

Hand out the shared private key (`aisb_key`) to every student, then assign each
student a pod IP + port from the `--list` output.

## 2. Pre-download models (GPU days only)

The full setup profile already runs `download_models.py`. Run it by hand only to
top up a pod's cache after adding a model — the API-only days need nothing here:

```bash
ssh root@<ip> -p <port> -i ~/.ssh/aisb_key \
    -o StrictHostKeyChecking=no -o PasswordAuthentication=no -o IdentitiesOnly=yes \
    python /workspace/aisb/3.1-tokenization/setup/download_models.py
```

The `--list` output above gives you the exact `ssh ...` prefix for each pod.

## 3. Verify the content still runs (`--test-solutions`)

Once the fleet is provisioned, check that the exercises actually execute on the
pods participants will get — real GPU, real installed dependencies:

```bash
# Day 1 (API-only), on every bootcamp pod
python runpod_setup/deploy_runpod.py --test-solutions --day 1 \
    --ssh-key ~/.ssh/aisb_key --git-private-key ~/.ssh/aisb_key \
    --openrouter-key sk-or-v1-xxx

# Every 3.x section, on every bootcamp pod
python runpod_setup/deploy_runpod.py --test-solutions --day 3 --ssh-key ~/.ssh/aisb_key

# Every section in the repo, on one pod
python runpod_setup/deploy_runpod.py --test-solutions --all --pod <pod_id> \
    --ssh-key ~/.ssh/aisb_key
```

Exactly one of `--day N` (repeatable) or `--all` is required.

`--openrouter-key` is needed for the API-only days (1, 2), whose exercises call
OpenRouter. It falls back to the `OPENROUTER_API_KEY` env var. The script writes a
temporary `/workspace/aisb/.env` holding the key on the pod for the run and deletes it
afterwards in a `finally`, so it does not outlive the test even if the run fails or
times out.

For a single API-only day, `--test-solutions` also installs that day's lightweight
dependencies first, so it works on any cloned pod regardless of how it was
provisioned. GPU days (3+) assume the pod already went through the full setup.
`--git-private-key` is not used by this path (the repo is already cloned); passing it
is harmless.

For each section this runs `aisb_utils/test_solutions.py` on the pod, which:

1. generates `sectionN_answers.py` from `sectionN_solution.py` with every
   `if "SOLUTION":` branch filled in and the `test_*` calls left in place;
2. executes it with the section folder as the working directory;
3. **deletes the answers file immediately**, pass or fail.

Step 3 matters: that file contains every answer for the section, and it runs on
participant pods. It is removed in a `finally` block, a stale-file sweep cleans
up after a hard kill, and `*_answers.py` is gitignored.

Pass `--remain` when running the script directly to keep the generated files, so
a failing section can be edited and re-run by hand. The next run sweeps them
away. Don't leave them sitting on a pod participants can reach.

Output streams live, and each pod ends with a per-section PASS/FAIL table plus
the tail of every failing traceback. The command exits non-zero if any pod had
any section fail. Pods run one at a time so the output stays attributable.

You can also run it directly on a pod, or locally for the API-only sections:

```bash
python aisb_utils/test_solutions.py --day 1
python aisb_utils/test_solutions.py --all --list     # dry run: list what would run
python aisb_utils/test_solutions.py --day 1 --remain # keep the files to debug
```

## 4. Deprovision

After the session, terminate all bootcamp pods in one go:

```bash
python runpod_setup/deploy_runpod.py --stop-all
```

Verify nothing is left running:

```bash
python runpod_setup/deploy_runpod.py --list
```
