# 5.1 — Model editing

## What to expect

Participants investigate a surgical factual edit as a training-time integrity
problem, measuring whether the change works and what unrelated behavior it harms.

**Suggested time:** 60 minutes

**Exercises:** [Open the participant instructions](section1_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | PyTorch model inspection | Run a reproducible model-editing harness and inspect the resulting parameter delta |
| ML | Transformer MLPs, hidden representations, and output logits | Explain the high-level ROME mechanism and measure efficacy, paraphrase generalization, and locality |
| Security | Model-artifact integrity and access control | Threat-model who can edit weights, how an edit could be introduced, and how defenders might detect it |
| Theory | - | Distinguish a successful targeted edit from evidence of a general, precise model-control capability |

### Background

- Revisit the transformer/MLP portions of Karpathy's [*Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI).
- Read the [PoisonGPT case study](https://blog.mithrilsecurity.io/poisongpt-how-we-hid-a-lobotomized-llm-on-hugging-face-to-spread-fake-news/) for the model-supply-chain threat motivating the exercise.
- Read the abstract and method overview of the [ROME paper](https://arxiv.org/abs/2202.05262); the full paper is optional longer reading because the exercise demonstrates the implementation.
