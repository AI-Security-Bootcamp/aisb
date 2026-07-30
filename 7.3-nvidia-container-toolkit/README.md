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

### Current-state TODOs

- [ ] Add an automated safe sanity check for the payload build that does not require the vulnerable runtime.
- [ ] Add a post-patch comparison proving that the exploit no longer crosses the host boundary.
- [ ] Time and verify the complete participant route on the course image.
