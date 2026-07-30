# 2.4 — From monitor scores to system safety

## What to expect

This optional section turns local monitor performance into a system-level Monte
Carlo model involving attack frequency, audit rate, latency, and repeated attack
opportunities.

**Suggested time:** 30 minutes (optional)

**Exercises:** [Open the participant instructions](section4_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | - | Record parameter sweeps and communicate results in a compact table or plot |
| ML | Probability, expected values, and sample frequencies | Interpret Monte Carlo estimates and distinguish them from measured deployment safety |
| Security | False positives, false negatives, audit rates, and attack prevalence | Identify which operational assumptions dominate system risk and where defense in depth enters |
| Theory | - | Explain why component AUC alone does not determine end-to-end safety |

### Background

- Complete [2.2](../2.2-monitoring/README.md) and preferably [2.3](../2.3-control-protocols/README.md).
