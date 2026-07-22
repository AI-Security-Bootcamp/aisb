# 4.1 — Model editing

## What to expect

Participants investigate a surgical factual edit as a training-time integrity
problem, measuring whether the change works and what unrelated behavior it harms.

**Suggested time:** 60 minutes

**Exercises:** [Open the participant instructions](section1_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | PyTorch model inspection, local-model inference, and notebook use | Run a reproducible model-editing harness and inspect the resulting parameter delta |
| ML | Transformer MLPs, representations, logits, and basic causal intervention intuition | Explain the high-level ROME mechanism and measure efficacy, paraphrase generalization, and locality |
| Security | Software/data integrity and unauthorized change detection | Threat-model who can edit weights, how an edit could be introduced, and how defenders might detect it |
| Theory | Facts as distributed model behavior rather than database rows | Distinguish a successful targeted edit from evidence of a general, precise model-control capability |

### Preparation

- Revisit the transformer/MLP portions of Karpathy's [*Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI).
- Read the [ROME paper overview](https://arxiv.org/abs/2202.05262); no prior ROME implementation should be assumed.

### Current-state TODOs

- [ ] Replace the external “play with this Colab” task with a pinned, local, reproducible harness or captured outputs.
- [ ] Require predictions and measurements for efficacy, paraphrases, neighborhood/locality, and unrelated capability regression.
- [ ] Inspect the size and location of the parameter delta and connect it to detection or provenance.
- [ ] Add an explicit attacker-access and model-deployment threat model.
