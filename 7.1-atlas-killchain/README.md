# 7.1 — ATLAS kill-chain warm-up

## What to expect

Participants use MITRE ATLAS vocabulary to convert isolated techniques into a
conditional attack path with explicit assets, trust boundaries, preconditions,
postconditions, and impact.

**Suggested time:** 30 minutes

**Exercises:** [Open the participant instructions](section1_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | System architecture and trust-boundary diagrams | Produce a concise architecture/trust-boundary view supporting the selected chain |
| ML | ML-system attack surfaces and attacker-access models | Place ML-specific techniques in the correct system context rather than treating the model as the whole target |
| Security | ATT&CK/ATLAS tactics and techniques; attack preconditions and postconditions | Build a plausible chain, justify skipped tactics, and distinguish prevention, detection, and response opportunities |
| Theory | Safety claims, evidence, assumptions, and catastrophic risk | Distinguish patchable defects from mission-inherent exposure without assuming governance applies only to the latter |

### Background

- Required prework: [MITRE ATLAS / *AI Security 101*](https://atlas.mitre.org/).
- Review the section READMEs for Days 1–6, especially their outgoing security and theory outcomes.

### Current-state TODOs

- [ ] Begin with target assets, trust boundaries, attacker access/budget, and objective before selecting techniques.
- [ ] Explain that tactic columns are useful categories, not a mandatory left-to-right sequence.
- [ ] Verify every cited ATLAS technique name/ID against the current primary matrix.
- [ ] Replace overly broad claims that governance work lives only in the “inherent property” category.
- [ ] Keep the deliverable small enough that reasoning quality matters more than matrix completion.
