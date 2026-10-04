# 1.5 — RAND report: securing AI model weights

## What to expect

Participants use RAND's operational-capability and security-level frameworks to
classify adversaries and justify controls for concrete model-weight environments.

**Suggested time:** 60 minutes

**Exercises:** [Open the participant instructions](section5_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | - | Identify the systems and environments included in a model-weight protection boundary |
| ML | Model weights, training, inference, and deployment environments | Explain why weights are a distinctive high-value asset |
| Security | Threat modeling and defense in depth | Match controls to an explicit adversary capability and state residual risk |
| Theory | - | Distinguish operational-capability tiers from security-level targets |

### Background

Read these parts of RAND's [*Securing AI Model Weights*](https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2800/RRA2849-1/RAND_RRA2849-1.pdf)
before the discussion. This is the required reading; the full report is optional.
Links jump to the first PDF page in each range. **Printed pages** are the numbers
shown on the report itself; **PDF pages** count from the cover.

| Priority | Printed pages | PDF pages | Focus |
| --- | --- | --- | --- |
| Read | [v–vi](https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2800/RRA2849-1/RAND_RRA2849-1.pdf#page=5) | 5–6 | Summary: why protect weights and the main recommendations |
| Read | [9–10](https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2800/RRA2849-1/RAND_RRA2849-1.pdf#page=19) | 19–20 | OC1–OC5 definitions, especially Figure 4.1 on p. 10: resources, access, and budgets |
| Read | [14–18](https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2800/RRA2849-1/RAND_RRA2849-1.pdf#page=24) | 24–28 | Attack feasibility: Table 5.2 on pp. 15–17 and its interpretation and uncertainty in Box 5.1 on pp. 17–18 |
| Read | [21–23](https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2800/RRA2849-1/RAND_RRA2849-1.pdf#page=31) | 31–33 | SL1–SL5 definitions in Figure 6.1 on p. 22 and the five environments on p. 23 |
| Skim | [24–32](https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2800/RRA2849-1/RAND_RRA2849-1.pdf#page=34) | 34–42 | Benchmark controls; focus on the changes from SL3 to SL4 to SL5 in Tables 6.3–6.5 and their accompanying takeaways |

- Optional deeper reference: [Appendix A, attack examples (p. 37)](https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2800/RRA2849-1/RAND_RRA2849-1.pdf#page=47), [Appendix B, detailed controls (p. 71)](https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2800/RRA2849-1/RAND_RRA2849-1.pdf#page=81), and [Appendix C, control comparison (p. 95)](https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2800/RRA2849-1/RAND_RRA2849-1.pdf#page=105).
- Optional supporting material: [RAND and IAPS discussion](optional-iaps-discussion.md).
