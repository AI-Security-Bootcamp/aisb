# 3.5 — Model weight extraction via SVD

## What to expect

Participants use black-box logit queries and SVD to infer hidden dimension and
extract the output projection up to an unknown linear transform.

**Suggested time:** 45 minutes for the core route; up to 60 additional minutes for reconstruction stretch work

**Exercises:** [Open the participant instructions](section5_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Array manipulation in NumPy | Build a query matrix, inspect singular values, and track query cost and numerical rank |
| ML | Given a matrix, identify its dimensions and rank; state the three factors returned by SVD; relate logits to a hidden vector times an output matrix | Explain why low-rank logits reveal hidden dimension and how SVD extracts the final projection up to a linear transform |
| Security | Define a black-box query interface and name how rate limits, top-k truncation, rounding, or noise constrain it | Relate returned logit detail and query volume to extraction feasibility and mitigations |
| Theory | - | Explain what “up to an unknown linear transform” means for extracted weights |

### Background

- Read the linked [SVD refresher](https://en.wikipedia.org/wiki/Singular_value_decomposition) and review change-of-basis intuition before the section.
- Revisit [1.2 on logprob/logit exposure](../1.2-logprobs/README.md); the instructor should provide an exact linear-algebra refresher if SVD is not required prework.

### Current-state TODOs

- [ ] Explain that the alignment transform is currently fitted using true white-box weights unavailable to the stated attacker.
- [ ] Wrap the local model as an actual oracle and meter query cost so the access model is credible.
- [ ] Make the 60-minute reconstruction optional or provide more numerical scaffolding relative to its learning value.
- [ ] Add economic and API-policy analysis: top-k outputs, noise, rounding, and rate limits.
