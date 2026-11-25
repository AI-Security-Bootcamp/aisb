# 3.5 — Model weight extraction via SVD

## What to expect

Participants use black-box logit queries and SVD to infer GPT-2's hidden
dimension and extract its output projection up to an unknown linear transform.

**Suggested time:** 45 minutes for 3.5.2; 10 minutes for optional 3.5.1;
60 minutes for 3.5.3.

**Exercises:** [Open the participant instructions](section5_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Stack NumPy vectors into a matrix and select its rows or columns by index | Build a query matrix, inspect singular values, and track query cost and numerical rank |
| ML | Given a hidden state of shape `(1, d)` and output weights of shape `(d, vocab_size)`, state the shape of the logits | Explain why low-rank logits reveal hidden dimension and how SVD extracts the final projection up to a linear transform |
| Security | Given an API response containing logits, distinguish it from sampled text and identify which output information is exposed | Relate returned logit detail and query volume to extraction feasibility and mitigations |
| Theory | - | Explain what “up to an unknown linear transform” means for extracted weights |

### Background

- NumPy's [`linalg.svd` reference](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html): return shapes and `compute_uv=False`.
- 3Blue1Brown's [change-of-basis lesson](https://www.3blue1brown.com/lessons/change-of-basis/).
- [1.2 on logprob/logit exposure](../1.2-logprobs/README.md).
- Optional longer reading: [SVD overview](https://en.wikipedia.org/wiki/Singular_value_decomposition).
