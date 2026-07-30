# 2.3 — AI-control protocols

## What to expect

Participants turn monitor scores into decisions by implementing trusted
monitoring, defer-to-trusted, and resampling policies, then compare their
safety–usefulness trade-offs.

**Suggested time:** 90 minutes

**Exercises:** [Open the participant instructions](section3_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | `async`/`await` | Implement stateful control-policy decisions and analyze evaluation logs |
| ML | Decision thresholds and stochastic sampling | Explain how thresholds, replacement models, and repeated samples change system behavior |
| Security | Audit budgets, escalation policies, and residual risk | Design a protocol around attacker behavior rather than treating monitoring as the final control |
| Theory | AI-control trust models and safety–usefulness trade-offs | Reason about the safety–usefulness frontier and the assumptions under which each protocol helps |

### Background

- Complete [2.2](../2.2-monitoring/README.md) first.
- Read Sections 3.2–3.3 of [*AI Control: Improving Safety Despite Intentional Subversion*](https://arxiv.org/abs/2312.06942) and the abstract and introduction of [*Ctrl-Z: Controlling AI Agents via Resampling*](https://arxiv.org/abs/2504.10374).
- References: Python's [`asyncio` coroutines and tasks](https://docs.python.org/3/library/asyncio-task.html) and the [ControlArena reference](https://control-arena.aisi.org.uk/reference/).

### Current-state TODOs

- [ ] Increase or precompute the evaluation sample so a 2% audit budget has statistical meaning; ten trajectories are insufficient.
- [ ] Make the policy decision itself, not private framework state, the main implementation challenge.
- [ ] Require interpretation of uncertainty and run-to-run variance in the final comparison.
- [ ] Give Exercise 2.3.4 a concrete written analysis deliverable.
