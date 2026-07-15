# 3.4 — Knowledge-distillation attacks

## What to expect

Participants retrieve a standard PyTorch training-loop pattern, implement it,
and extend it with temperature-scaled distillation to investigate how teacher
outputs transfer information.

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Required PyTorch Blitz: tensors, modules, optimizers, `zero_grad`, `backward`, and `step` | Implement and debug supervised and distillation training steps with useful diagnostics |
| ML | Logits, softmax, cross-entropy, masking with `ignore_index`, and introductory KL divergence | Implement temperature-scaled KD and identify which loss terms create which gradients |
| Security | API observability and extraction threat models | Explain when teacher logits, probabilities, or labels enable capability transfer and what access an attacker needs |
| Theory | Student/teacher models and train/eval controls | Design an experiment that distinguishes target masking, example removal, and knowledge transfer |

### Preparation

- Required prework: *Deep Learning with PyTorch: A 60 Minute Blitz*.
- Just-in-time references: PyTorch's training-loop example and the documented behavior of `torch.nn.functional.kl_div`.

### Current-state TODOs

- [ ] Keep learner implementation of the training loop, but split it into observable checkpoints before adding KD.
- [ ] Replace the current corpus with a controlled comparison: remove examples, mask targets, omit KD positions, and distil normally.
- [ ] The current corpus contains the forbidden association in model inputs and additional France sentences, so it does not support the stated omission claim.
- [x] State the KL direction implemented by `F.kl_div` precisely and avoid claiming softmax probability is exactly zero.
- [ ] Distinguish legitimate internal compression from API-based model-stealing threat models.
