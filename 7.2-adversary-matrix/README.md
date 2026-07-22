# 7.2 — Frontier-lab adversary matrix and defensive plan

## What to expect

Participants synthesize the week into one credible frontier-lab threat path,
identify where controls are strong or thin, and propose a prioritized defensive
or research response.

**Suggested time:** 90 minutes

**Exercises:** [Open the participant instructions](section2_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Draw an architecture containing application, training, cluster, credential, and model-artifact components and mark their trust boundaries | Connect those components in one defensible end-to-end narrative |
| ML | Name one attack, one defense, and the required attacker access for each technical day followed | Identify which model capabilities or interfaces each attack step depends on and avoid importing untaught mechanisms |
| Security | Build a five-step kill chain, map each step to ATT&CK/ATLAS, and state one control plus residual risk | Validate preconditions between steps, prioritize chain-breaking controls, and identify detection gaps |
| Theory | Write a safety claim with evidence, assumptions, and one prioritized action | Translate a threat model into claims, evidence needs, assumptions, and a concrete next action |

### Background

- Complete [7.1](../7.1-atlas-killchain/README.md) and skim the READMEs in each numbered section folder.
- Use [MITRE ATLAS](https://atlas.mitre.org/) and [MITRE ATT&CK](https://attack.mitre.org/) as references, not as forms that must be filled exhaustively.

### Current-state TODOs

- [ ] Reduce the requested 2–3 entries across 13 columns to one high-quality 6–8-step path plus justified gaps.
- [ ] Require a prioritized control/research plan as the final artifact, not only a populated matrix.
- [x] Correct cross-references to material not actually taught, including residual VRAM and firmware flashing.
- [x] Reclassify watermarking as a provenance/defense mechanism rather than an attacker collection technique.
- [x] Remove the claim that the current Day 6 simulator demonstrates access to a neighboring tenant's memory.
- [ ] Keep the capstone usable for attendees who followed the core route rather than every stretch exercise.
