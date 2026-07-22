# 2.2 — Coding-agent attack surface and affordances

## What to expect

Participants apply their existing systems-security experience to a coding agent,
mapping compromised inputs to concrete actions, impacts, and layered controls.

**Suggested time:** 30–40 minutes

**Exercises:** [Open the participant instructions](section2_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Trace how a repository action reaches a shell, network, or CI runner and list the credentials available at each step | Describe the effective authority created by file, shell, network, MCP, and CI access |
| ML | Draw an agent loop containing a model response, tool call, tool result, and next model response | Distinguish model capability from the affordances granted by the surrounding agent harness |
| Security | Produce a threat model with a named asset, actor, trust boundary, and least-privilege control | Build an affordance → exploit → impact → mitigation map and identify controls that only displace risk |
| Theory | - | Relate prompt-injected agents, misaligned agents, monitor manipulation, and rogue deployments without conflating them |

### Background

- Required prework: Anthropic's [*How Claude Code works*](https://code.claude.com/docs/en/how-claude-code-works) and Google's [MCP overview](https://cloud.google.com/discover/what-is-model-context-protocol).
- Bring experience from a developer environment or CI/CD system you have secured.

### Current-state TODOs

- [x] Remove the cargo-cult code setup from this prose-only section.
- [ ] Anchor the exercise in one concrete agent deployment with named assets and trust boundaries.
- [ ] Ask pairs to prioritize a small number of attack paths and controls instead of producing an unbounded catalogue.
- [ ] Replace or source volatile adoption claims that do not contribute to the exercise.
