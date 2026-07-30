# 2.1 — Coding-agent attack surface and affordances

## What to expect

Participants apply their existing systems-security experience to a coding agent,
mapping compromised inputs to concrete actions, impacts, and layered controls.

**Suggested time:** 30–40 minutes

**Exercises:** [Open the participant instructions](section1_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Repository automation, shells, CI, and credential exposure | Describe the effective authority created by file, shell, network, MCP, and CI access |
| ML | LLM agent and tool-use loops | Distinguish model capability from the affordances granted by the surrounding agent harness |
| Security | Threat modeling and least privilege | Build an affordance → exploit → impact → mitigation map and identify controls that only displace risk |
| Theory | - | Relate prompt-injected agents, misaligned agents, monitor manipulation, and rogue deployments without conflating them |

### Background

- Required prework: Anthropic's [*How Claude Code works*](https://code.claude.com/docs/en/how-claude-code-works) and Google's [MCP overview](https://cloud.google.com/discover/what-is-model-context-protocol).

### Current-state TODOs

- [x] Remove the cargo-cult code setup from this prose-only section.
- [ ] Anchor the exercise in one concrete agent deployment with named assets and trust boundaries.
- [ ] Ask pairs to prioritize a small number of attack paths and controls instead of producing an unbounded catalogue.
- [ ] Replace or source volatile adoption claims that do not contribute to the exercise.
