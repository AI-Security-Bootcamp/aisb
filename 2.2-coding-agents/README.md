# 2.2 — Coding-agent attack surface and affordances

## What to expect

Participants apply their existing systems-security experience to a coding agent,
mapping compromised inputs to concrete actions, impacts, and layered controls.

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Familiarity with repositories, shells, CI/CD, credentials, and developer tooling | Describe the effective authority created by file, shell, network, MCP, and CI access |
| ML | High-level understanding of coding agents from the Anthropic pre-reading | Distinguish model capability from the affordances granted by the surrounding agent harness |
| Security | Threat modeling, least privilege, persistence, supply-chain security, and monitoring | Build an affordance → exploit → impact → mitigation map and identify controls that only displace risk |
| Theory | Basic AI-control threat categories | Relate prompt-injected agents, misaligned agents, monitor manipulation, and rogue deployments without conflating them |

### Preparation

- Required prework: Anthropic's *How Claude Code works* and the MCP overview.
- Bring experience from a developer environment or CI/CD system you have secured.

### Current-state TODOs

- [x] Remove the cargo-cult code setup from this prose-only section.
- [ ] Anchor the exercise in one concrete agent deployment with named assets and trust boundaries.
- [ ] Ask pairs to prioritize a small number of attack paths and controls instead of producing an unbounded catalogue.
- [ ] Replace or source volatile adoption claims that do not contribute to the exercise.
