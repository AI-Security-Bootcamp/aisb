# 3.4 — Knowledge-distillation attacks

## What to expect

Participants retrieve a standard PyTorch training-loop pattern, implement it,
and extend it with temperature-scaled distillation to investigate how teacher
outputs transfer information.

**Suggested time:** 90 minutes

**Exercises:** [Open the participant instructions](section4_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | PyTorch training loops | Implement and debug supervised and distillation training steps with useful diagnostics |
| ML | Logits, softmax, cross-entropy, `ignore_index`, and KL divergence | Implement temperature-scaled KD and identify which loss terms create which gradients |
| Security | Model access levels: labels, probabilities, logits, activations, and weights | Explain when teacher logits, probabilities, or labels enable capability transfer and what access an attacker needs |
| Theory | Teacher/student models, train/eval modes, and held-out evaluation | Design an experiment that distinguishes target masking, example removal, and knowledge transfer |

### Background

- Required prework: [*Deep Learning with PyTorch: A 60 Minute Blitz*](https://docs.pytorch.org/tutorials/beginner/blitz/index.html).
- References: PyTorch's [optimization-loop tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) and the documented behavior of [`torch.nn.functional.kl_div`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html).
- PyTorch's full [knowledge-distillation tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html) is an optional worked example because it substantially overlaps this section.
