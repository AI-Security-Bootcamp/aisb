# 3.4 — Knowledge-distillation attacks

## What to expect

Participants retrieve a standard PyTorch training-loop pattern, implement it,
and extend it with temperature-scaled distillation to investigate how teacher
outputs transfer information.

**Suggested time:** 90 minutes

**Exercises:** [Open the participant instructions](section4_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Given a module, batch, and optimizer, write `zero_grad` → forward pass → loss → `backward` → `step`, then print a scalar loss | Implement and debug supervised and distillation training steps with useful diagnostics |
| ML | Explain logits, softmax, cross-entropy, `ignore_index`, and what a KL divergence compares | Implement temperature-scaled KD and identify which loss terms create which gradients |
| Security | Distinguish label-only, probability, logit, activation, and weight access to a model | Explain when teacher logits, probabilities, or labels enable capability transfer and what access an attacker needs |
| Theory | Explain teacher/student roles, switch a model between train/eval modes, and state why evaluation data is held out | Design an experiment that distinguishes target masking, example removal, and knowledge transfer |

### Background

- Required prework: [*Deep Learning with PyTorch: A 60 Minute Blitz*](https://docs.pytorch.org/tutorials/beginner/blitz/index.html).
- Just-in-time references: PyTorch's [optimization-loop tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) and the documented behavior of [`torch.nn.functional.kl_div`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html).

### Current-state TODOs

- [ ] Keep learner implementation of the training loop, but split it into observable checkpoints before adding KD.
- [ ] Replace the current corpus with a controlled comparison: remove examples, mask targets, omit KD positions, and distil normally.
- [ ] The current corpus contains the forbidden association in model inputs and additional France sentences, so it does not support the stated omission claim.
- [x] State the KL direction implemented by `F.kl_div` precisely and avoid claiming softmax probability is exactly zero.
- [ ] Distinguish legitimate internal compression from API-based model-stealing threat models.
