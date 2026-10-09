# 3.5 — Model weight extraction via SVD

## What to expect

Participants recover the hidden dimension of a toy matrix product with SVD,
then do the same with black-box logit queries from GPT-2 and extract its output
projection up to an unknown linear transform.

**Suggested time:** 55 minutes for the core route; up to 60 additional minutes for reconstruction stretch work

**Exercises:** [Open the participant instructions](section5_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Array manipulation in NumPy | Build a query matrix, inspect singular values, and track query cost and numerical rank |
| ML | Given `A` of shape `(n, d)` and `B` of shape `(d, m)`, state the shape of `A @ B` | Find the rank of a low-rank matrix product from its singular values; explain why low-rank logits reveal hidden dimension and how SVD extracts the final projection up to a linear transform |
| Security | Black-box model access, rate limits, top-k truncation, rounding, and output noise | Relate returned logit detail and query volume to extraction feasibility and mitigations |
| Theory | - | Explain what “up to an unknown linear transform” means for extracted weights |

### Background

- Read the [SVD refresher](https://en.wikipedia.org/wiki/Singular_value_decomposition) and 3Blue1Brown's [change-of-basis lesson](https://www.3blue1brown.com/lessons/change-of-basis/) before the section.
- Revisit [1.2 on logprob/logit exposure](../1.2-logprobs/README.md) for the API access model used by the attack.
