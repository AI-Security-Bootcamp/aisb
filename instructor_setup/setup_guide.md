# Instructor Setup Guide

Step-by-step checklist for provisioning the bootcamp infrastructure. Each day
uses a different compute backend; the table below summarises which script drives
which day.

| Day | Sections | Backend | Script |
| --- | --- | --- | --- |
| 1 | 1.1 – 1.4 | RunPod (API-only, lightweight) | `deploy_runpod.py` |
| 2 | 2.1 – 2.4 | GCP VMs (API-only) | `day2_setup/setup_day2.py` |
| 3 | 3.1 – 3.5 | RunPod (GPU) | `deploy_runpod.py` |
| 5 | 5.1 – 5.4 | RunPod (GPU) | `deploy_runpod.py` |
| 6 | 6.1 – 6.4 | RunPod (GPU) | `deploy_runpod.py` |
| 7 | 7.1 – 7.4 | GCP VMs (manual) | manual SSH setup |

> **Days 1 and 2 must not share a pod.** Their dependency stacks conflict. Deploy
> separate fleets, or re-provision between days.

---

## 0. Prerequisites

### 0.1 API keys

Create a `.env` file in the repo root (it is gitignored):

```
OPENROUTER_API_KEY=sk-or-v1-...
RUNPOD_API_KEY=...
LAMBDA_API_KEY=...          # only if using Lambda Cloud
HF_TOKEN=hf_...             # needed for Day 5 gated models
```

### 0.2 SSH key

Generate **one key** for everything — pod access, VM access, and cloning the
private repo onto each machine. Use an empty passphrase (`-N ""`) because the
scripts drive hosts non-interactively.

```bash
ssh-keygen -t ed25519 -C "aisb_key" -f ~/.ssh/aisb_key -N ""
```

Then register `~/.ssh/aisb_key.pub` as a **read-only deploy key** on the GitHub
repo (*Settings → Deploy keys*, leave *Allow write access* unchecked). Revoke it
once the cohort ends.

> **Security note:** participants have root on their pods and can read the deploy
> key. Keep it read-only and short-lived.

### 0.3 Instance list (Day 2 only)

Day 2 runs on pre-provisioned GCP VMs. List their addresses in
`instructor_setup/day2_setup/instance.txt`, one per line as `user@host`:

```
participant@34.60.180.171
participant@35.188.39.150
...
```

---

## 1. Day 1 — RunPod (API-only)

### Deploy

```bash
python instructor_setup/deploy_runpod.py \
    --count 12 --day 1 \
    --ssh-key ~/.ssh/aisb_key \
    --git-private-key ~/.ssh/aisb_key
```

### Verify

```bash
python instructor_setup/deploy_runpod.py \
    --test-solutions --day 1 \
    --ssh-key ~/.ssh/aisb_key \
    --git-private-key ~/.ssh/aisb_key \
    --openrouter-key sk-or-v1-xxx
```

### List pods and hand out connection info

```bash
python instructor_setup/deploy_runpod.py --list --ssh-key ~/.ssh/aisb_key
```

---

## 2. Day 2 — GCP VMs

Day 2 uses its own script because it targets a fleet of GCP VMs listed in
`instance.txt` rather than RunPod pods.

### Deploy (clone repo, install deps, write .env)

```bash
python instructor_setup/day2_setup/setup_day2.py \
    --ssh-key ~/.ssh/aisb_key \
    --git-private-key ~/.ssh/aisb_key
```

### Verify (smoke test — catches broken imports/missing keys in a few minutes)

```bash
python instructor_setup/day2_setup/setup_day2.py \
    --test-solutions \
    --ssh-key ~/.ssh/aisb_key \
    --git-private-key ~/.ssh/aisb_key \
    --openrouter-key sk-or-v1-xxx
```

### Verify (detailed — runs every section to completion, ~30+ min)

Use `--detail` before a cohort or after changing a Day 2 solution file. Section
2.3 alone takes over half an hour.

```bash
python instructor_setup/day2_setup/setup_day2.py \
    --test-solutions --detail \
    --ssh-key ~/.ssh/aisb_key \
    --git-private-key ~/.ssh/aisb_key \
    --openrouter-key sk-or-v1-xxx
```

### SSH to a specific VM

```bash
ssh participant@34.60.180.171 -p 22 \
    -i ~/.ssh/aisb_key \
    -o StrictHostKeyChecking=no \
    -o PasswordAuthentication=no \
    -o IdentitiesOnly=yes
```

---

## 3. Day 3 — RunPod (GPU)

### Deploy

```bash
python instructor_setup/deploy_runpod.py \
    --count 12 --day 3 \
    --ssh-key ~/.ssh/aisb_key \
    --git-private-key ~/.ssh/aisb_key
```

Omitting `--day` (or passing `--day 3`) triggers the full GPU setup profile:
pinned deps, GPU check, model downloads. This takes several minutes per pod.

### Verify (single pod)

```bash
python instructor_setup/deploy_runpod.py \
    --test-solutions --day 3 \
    --ssh-key ~/.ssh/aisb_key \
    --git-private-key ~/.ssh/aisb_key \
    --pod <POD_ID> \
    --skip 3.3.5
```

### Verify (all pods in parallel)

```bash
python instructor_setup/deploy_runpod.py \
    --test-solutions --day 3 --parallel \
    --ssh-key ~/.ssh/aisb_key \
    --git-private-key ~/.ssh/aisb_key \
    --skip 3.3.5
```

---

## 4. Day 5 — RunPod (GPU)

Re-use the GPU fleet from Day 3 or deploy a fresh one. Day 5 has long-running
fine-tuning sections; raise `--section-timeout` accordingly.

### Deploy (if needed)

```bash
python instructor_setup/deploy_runpod.py \
    --count 12 --day 5 \
    --ssh-key ~/.ssh/aisb_key \
    --git-private-key ~/.ssh/aisb_key
```

### Verify (all pods, skipping 5.4)

```bash
python instructor_setup/deploy_runpod.py \
    --test-solutions --day 5 --parallel \
    --ssh-key ~/.ssh/aisb_key \
    --git-private-key ~/.ssh/aisb_key \
    --hf-token hf_xxx \
    --section-timeout 5400 \
    --skip 5.4
```

### Verify (single pod)

```bash
python instructor_setup/deploy_runpod.py \
    --test-solutions --day 5 \
    --ssh-key ~/.ssh/aisb_key \
    --git-private-key ~/.ssh/aisb_key \
    --hf-token hf_xxx \
    --section-timeout 5400 \
    --skip 5.4 \
    --pod <POD_ID>
```

---

## 5. Day 6 — RunPod (GPU)

### Deploy (if needed)

```bash
python instructor_setup/deploy_runpod.py \
    --count 12 --day 6 \
    --ssh-key ~/.ssh/aisb_key \
    --git-private-key ~/.ssh/aisb_key
```

### Verify

```bash
python instructor_setup/deploy_runpod.py \
    --test-solutions --day 6 --parallel \
    --ssh-key ~/.ssh/aisb_key \
    --git-private-key ~/.ssh/aisb_key
```

---

## 6. Day 7 — GCP VMs (manual)

Day 7 runs on dedicated GCP VMs that need a specific vulnerable software stack
(NVIDIA Container Toolkit 1.17.7, pinned runc, etc.). These are set up manually
rather than through the automated scripts.

### SSH to a Day 7 VM

```bash
ssh ubuntu@<VM_IP> \
    -i ~/.ssh/aisb_key \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o PasswordAuthentication=no \
    -o IdentitiesOnly=yes
```

### Provision

Once connected, run the setup script from within the repo:

```bash
cd /path/to/aisb/7.3-nvidia-container-toolkit/module1
sudo bash setup.txt
```

See `7.3-nvidia-container-toolkit/module1/setup.txt` for the full known-good
stack (Ubuntu 22.04, NVIDIA A10, driver 580, Docker 29, runc 1.1.0).

---

## 7. Common operations

### Re-provision pods after an interrupted deploy

Don't re-run `--count` — that creates a second fleet. Provision the existing
pods instead:

```bash
python instructor_setup/deploy_runpod.py \
    --provision --day <N> \
    --ssh-key ~/.ssh/aisb_key \
    --git-private-key ~/.ssh/aisb_key
```

Or target specific pods:

```bash
python instructor_setup/deploy_runpod.py \
    --provision --day <N> \
    --pod <POD_ID> --pod <POD_ID> \
    --ssh-key ~/.ssh/aisb_key \
    --git-private-key ~/.ssh/aisb_key
```

### Replace a pod on a maintenance host

```bash
python instructor_setup/deploy_runpod.py --stop <POD_ID>
python instructor_setup/deploy_runpod.py \
    --count 1 --name aisb-bootcamp-<NN> --day <N> \
    --ssh-key ~/.ssh/aisb_key \
    --git-private-key ~/.ssh/aisb_key
```

### Tear down everything

```bash
# RunPod
python instructor_setup/deploy_runpod.py --stop-all
python instructor_setup/deploy_runpod.py --list   # verify nothing left

# Revoke the GitHub deploy key
```

---

## Quick-reference: flag cheat sheet

| Flag | Purpose |
| --- | --- |
| `--count N` | Create N new pods |
| `--day N` | Setup profile (1/2 = lightweight API-only; 3+ = full GPU) |
| `--ssh-key` | Private key for SSH access to pods/VMs |
| `--git-private-key` | Private key installed on pods to clone the repo |
| `--list` | List running pods with connection info |
| `--provision` | Re-provision existing pods (idempotent) |
| `--pod <ID>` | Target a specific pod (repeatable) |
| `--test-solutions` | Run reference solutions and report pass/fail |
| `--parallel` / `--no-parallel` | Run test pods concurrently or sequentially |
| `--skip <ID>` | Skip a section or exercise (e.g. `--skip 3.3.5`) |
| `--section-timeout N` | Per-section timeout in seconds (default 1800) |
| `--openrouter-key` | OpenRouter API key for API-only days |
| `--hf-token` | HuggingFace token for gated models |
| `--remain` | Keep generated answer files for debugging |
| `--stop <ID>` | Terminate one pod |
| `--stop-all` | Terminate all bootcamp pods |
| `--detail` | (Day 2 only) Full run instead of smoke test |
