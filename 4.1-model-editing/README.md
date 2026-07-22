# 4.1 — Model editing

## What to expect

Participants investigate a surgical factual edit as a training-time integrity
problem, measuring whether the change works and what unrelated behavior it harms.

**Suggested time:** 60 minutes

**Exercises:** [Open the participant instructions](section1_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Load a PyTorch model, enumerate named parameters, run one inference, and compare two tensors for changed values | Run a reproducible model-editing harness and inspect the resulting parameter delta |
| ML | Locate an MLP block in a transformer diagram and define hidden representation and output logits | Explain the high-level ROME mechanism and measure efficacy, paraphrase generalization, and locality |
| Security | Given a model artifact, identify who can modify it and name one integrity check that would detect a change | Threat-model who can edit weights, how an edit could be introduced, and how defenders might detect it |
| Theory | - | Distinguish a successful targeted edit from evidence of a general, precise model-control capability |

### Background

- Revisit the transformer/MLP portions of Karpathy's [*Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI).
- Read the abstract and method overview of the [ROME paper](https://arxiv.org/abs/2202.05262); the full paper is optional longer reading because the exercise demonstrates the implementation.

### Current-state TODOs

- [ ] Replace the external “play with this Colab” task with a pinned, local, reproducible harness or captured outputs.
- [ ] Require predictions and measurements for efficacy, paraphrases, neighborhood/locality, and unrelated capability regression.
- [ ] Inspect the size and location of the parameter delta and connect it to detection or provenance.
- [ ] Add an explicit attacker-access and model-deployment threat model.
