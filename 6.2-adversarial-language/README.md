# 6.2 — Discrete adversarial optimization with GCG

## What to expect

Participants adapt gradient-based optimization to discrete token sequences by
scoring target continuations, ranking token replacements, and running a greedy
coordinate search.

**Suggested time:** 75 minutes

**Exercises:** [Open the participant instructions](section2_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | PyTorch autograd and tensor indexing | Align target-token loss, extract embedding gradients, filter candidates, and implement a coordinate-update loop |
| ML | Token embeddings, causal language-model loss, and discrete optimization | Explain how gradients can guide token proposals even though tokens are discrete |
| Security | White-box attack models and adversarial evaluation | State required model access, measure attack success across restarts/prompts, and test transferability |
| Theory | - | Distinguish reducing target loss from demonstrating a robust jailbreak |

### Background

- Complete [6.1](../6.1-adversarial-vision/README.md) or refresh PyTorch [autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) and [optimization loops](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).
- Read the abstract and GCG method in [*Universal and Transferable Adversarial Attacks on Aligned Language Models*](https://arxiv.org/abs/2307.15043); the full paper is optional longer reading.

### Current-state TODOs

- [ ] Use harmless target continuations by default while preserving the refusal-bypass learning objective.
- [ ] Add tokenizer round-trip/printability constraints so optimized suffixes are deployable text.
- [ ] Evaluate actual completion behavior, ASR across restarts and held-out prompts, and transfer, not only loss reduction.
- [ ] Contrast white-box GCG with black-box and transfer-only attacker access.
