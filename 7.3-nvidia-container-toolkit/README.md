# 7.3 — NVIDIA Container Toolkit vulnerability

## What to expect

Participants reproduce CVE-2025-23266 in a disposable course environment and
trace how attacker-controlled container state reaches a privileged host hook.

**Suggested time:** 45 minutes

**Exercises:** [Open the participant instructions](section3_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Linux shared libraries, Docker images, and Make | Build an `LD_PRELOAD` payload and malicious test image |
| ML | GPU-enabled container deployment | Explain what the NVIDIA runtime adds to an ordinary container launch |
| Security | Linux privilege boundaries and container isolation | Trace the escape, identify the violated trust boundary, and select mitigations |
| Theory | - | Distinguish a patchable runtime vulnerability from inherent shared-infrastructure risk |

### Background

- Read the [NVIDIAScape analysis](https://www.wiz.io/blog/nvidia-ai-vulnerability-cve-2025-23266-nvidiascape).
- Use only the provided disposable lab host with the intentionally vulnerable toolkit configuration.

## Lab host version contract

This exercise depends on the container image's `LD_PRELOAD` environment variable
being inherited by the privileged NVIDIA `createContainer` hook. The following
stack was verified to reproduce the lab:

| Component | Version |
| --- | --- |
| OS | Ubuntu 22.04, Linux 5.15 |
| GPU / driver | NVIDIA A10, driver 580.173.02 |
| Docker | 29.1.3-0ubuntu3~22.04.2 |
| containerd | 2.2.1-0ubuntu1~22.04.2 |
| runc | 1.1.0-0ubuntu1 |
| NVIDIA Container Toolkit | 1.17.7-1 |
| libnvidia-container | 1.17.7-1 |
| NVIDIA runtime mode | `legacy` |
| CUDA compat mode | `hook` |

