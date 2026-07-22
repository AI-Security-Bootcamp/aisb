# 2.5 — From monitor scores to system safety

## What to expect

This optional section turns local monitor performance into a system-level Monte
Carlo model involving attack frequency, audit rate, latency, and repeated attack
opportunities.

**Suggested time:** 30 minutes (optional)

**Exercises:** [Open the participant instructions](section5_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Ability to run the supplied interactive simulation | Record parameter sweeps and communicate results in a compact table or plot |
| ML | Elementary probability and sampling | Interpret Monte Carlo estimates and distinguish them from measured deployment safety |
| Security | Familiarity with monitor errors and audit constraints | Identify which operational assumptions dominate system risk and where defense in depth enters |
| Theory | Completion of the Day 2 control sections | Explain why component AUC alone does not determine end-to-end safety |

### Preparation

- Complete [2.3](../2.3-monitoring/README.md) and preferably [2.4](../2.4-control-protocols/README.md).
- No additional library preparation should be required.

### Current-state TODOs

- [x] Remove the unused code-answer setup.
- [ ] Ask learners to predict outcomes before running the simulator and submit a small results table.
- [ ] Document the origin and limitations of default parameters.
- [ ] Move the Day 2 summary out of this optional section.
