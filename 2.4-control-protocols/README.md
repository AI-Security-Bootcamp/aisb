# 2.4 — AI-control protocols

## What to expect

Participants turn monitor scores into decisions by implementing trusted
monitoring, defer-to-trusted, and resampling policies, then compare their
safety–usefulness trade-offs.

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Completion of 2.3, Python async basics, and willingness to inspect a new agent API | Implement stateful control-policy decisions and analyze evaluation logs |
| ML | Scores, thresholds, and sampling | Explain how thresholds, replacement models, and repeated samples change system behavior |
| Security | Audit budgets, escalation policies, detection limits, and defense in depth | Design a protocol around attacker behavior rather than treating monitoring as the final control |
| Theory | Trusted/untrusted model assumptions from AI control | Reason about the safety–usefulness frontier and the assumptions under which each protocol helps |

### Preparation

- Complete 2.3 first.
- Read the assigned excerpts from *AI Control: Improving Safety Despite Intentional Subversion*; use the supplied framework examples as the API reference.

### Current-state TODOs

- [ ] Increase or precompute the evaluation sample so a 2% audit budget has statistical meaning; ten trajectories are insufficient.
- [ ] Make the policy decision itself—not private framework state—the main implementation challenge.
- [ ] Require interpretation of uncertainty and run-to-run variance in the final comparison.
- [ ] Give Exercise 4.4 a concrete written analysis deliverable.
