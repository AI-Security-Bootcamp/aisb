# 3.5 — Model weight extraction via SVD

## What to expect

Participants use black-box logit queries and SVD to infer GPT-2's hidden
dimension and extract its output projection up to an unknown linear transform.
Exercise 3.5.1 offers an optional toy matrix example; skip directly to 3.5.2
if you can already read a singular-value spectrum.

**Suggested time:** 45 minutes for the core route (3.5.2), plus 10 minutes for
the optional warmup (3.5.1); up to 60 additional minutes for reconstruction (3.5.3).

**Exercises:** [Open the participant instructions](section5_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Stack NumPy vectors into a matrix and select its rows or columns by index | Build a query matrix, inspect singular values, and track query cost and numerical rank |
| ML | Given a hidden state of shape `(1, d)` and output weights of shape `(d, vocab_size)`, state the shape of the logits | Explain why low-rank logits reveal hidden dimension and how SVD extracts the final projection up to a linear transform; optionally recover rank from a toy matrix product |
| Security | Given an API response containing logits, distinguish it from sampled text and identify which output information is exposed | Relate returned logit detail and query volume to extraction feasibility and mitigations |
| Theory | - | Explain what “up to an unknown linear transform” means for extracted weights |

### Background

- Read NumPy's [`linalg.svd` reference](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html), focusing on the returned shapes and `compute_uv=False`, and review 3Blue1Brown's [change-of-basis lesson](https://www.3blue1brown.com/lessons/change-of-basis/) before the section. A compact rank/logit reference is included in Exercise 3.5.2, so skipping 3.5.1 does not remove required background.
- Revisit [1.2 on logprob/logit exposure](../1.2-logprobs/README.md) for the API access model used by the attack.
- Optional longer reading: the [SVD overview](https://en.wikipedia.org/wiki/Singular_value_decomposition), which expands on the rank and basis concepts used in the lab.
