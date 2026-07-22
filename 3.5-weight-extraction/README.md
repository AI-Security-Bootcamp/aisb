# 3.5 — Logit-space dimension and subspace extraction

## What to expect

Participants use black-box logit queries and SVD to infer hidden dimension and
recover the output projection's column space up to an unknown linear transform.

**Suggested time:** 45 minutes for the core route; up to 60 additional minutes for reconstruction stretch work

**Exercises:** [Open the participant instructions](section5_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | NumPy arrays, batching API queries, and basic numerical debugging | Build a query matrix, inspect singular values, and track query cost and numerical rank |
| ML | Matrix rank, linear maps, SVD, logits, and hidden states | Explain why low-rank logits reveal hidden dimension and what an unidentified basis does not recover |
| Security | Black-box model-extraction threat modeling | Relate returned logit detail and query volume to extraction feasibility and mitigations |
| Theory | Identifiability and equivalence up to a transform | Distinguish recovering a subspace from recovering independently usable model weights |

### Preparation

- Review [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) and change-of-basis intuition before the section.
- Revisit [1.2 on logprob/logit exposure](../1.2-logprobs/README.md); the instructor should provide an exact linear-algebra refresher if SVD is not required prework.

### Current-state TODOs

- [x] Rename the core result as output-subspace recovery rather than exact weight extraction.
- [ ] Explain that the alignment transform is currently fitted using true white-box weights unavailable to the stated attacker.
- [ ] Wrap the local model as an actual oracle and meter query cost so the access model is credible.
- [ ] Make the 60-minute reconstruction optional or provide more numerical scaffolding relative to its learning value.
- [ ] Add economic and API-policy analysis: top-k outputs, noise, rounding, and rate limits.
