# Section 3.1 pod provisioning

Instructor workflow for provisioning RunPod GPU pods for the inference sections (3.1–3.5).

## Prerequisites

- `RUNPOD_API_KEY` set in the repo-root `.env` file (or exported in the shell).
- A single SSH key pair for participant access (e.g. `id_ed25519` / `id_ed25519.pub`), shared across all students. The public key is injected into every pod; the same private key is handed out to every student.
- A git deploy key with read access to the bootcamp repo (a separate key, kept by instructors only).

## 1. Create pods in bulk

Deploy all pods in a single command — `--count` creates N pods, all sharing the same injected public key. `--git-private-key` clones the repo into `/workspace/aisb-sg` on each pod during startup.

```bash
python 3.1-tokenization/setup/deploy_runpod.py --ssh-key ./id_ed25519 --git-private-key ./id_ed25519 --count <N_PAIRS>
```

List pods (with SSH/Jupyter connection strings) at any time:

```bash
python 3.1-tokenization/setup/deploy_runpod.py --list --ssh-key ./id_ed25519
```

Hand out the shared private key (`id_ed25519`) to every student, then assign each student a pod IP + port from the `--list` output.

## 2. Pre-download models (optional but recommended)

Downloading the models the first time takes several minutes. Run the download script on each pod ahead of time so students don't wait:

```bash
ssh root@<ip> -p <port> -i ./id_ed25519 \
    -o StrictHostKeyChecking=no -o PasswordAuthentication=no -o IdentitiesOnly=yes \
    python /workspace/aisb-sg/3.1-tokenization/setup/download_models.py
```

The `--list` output above gives you the exact `ssh ...` prefix for each pod.

## 3. Deprovision

After the session, terminate all bootcamp pods in one go:

```bash
python 3.1-tokenization/setup/deploy_runpod.py --stop-all
```

Verify nothing is left running:

```bash
python 3.1-tokenization/setup/deploy_runpod.py --list
```
