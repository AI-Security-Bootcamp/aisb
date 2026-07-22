# 5.3 — Continuous adversarial prefixes

## What to expect

Participants optimize a shared prefix directly in embedding space, compare that
continuous relaxation with GCG, and examine why a strong optimization result may
not constitute a deployable API attack.

**Suggested time:** 30 minutes

**Exercises:** [Open the participant instructions](section3_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Complete 5.2, create a trainable tensor, update it with Adam, and retrieve token embeddings from a model | Construct `inputs_embeds`, optimize a shared latent tensor, and evaluate it on held-out prompts |
| ML | Average a loss across a batch and explain the difference between an embedding vector and a discrete token ID | Explain why latent optimization is easier than discrete token search and measure transfer across examples |
| Security | Given an inference interface, state whether the caller can submit token IDs, embeddings, or text only | Identify the narrow settings in which an attacker can supply embeddings and avoid presenting the result as an ordinary API jailbreak |
| Theory | Define a continuous relaxation and explain why it can provide an upper-bound experiment | Use latent-prefix success as evidence about optimization/capability, not automatically about real-world exploitability |

### Background

- Complete [5.2](../5.2-adversarial-language/README.md) first.
- Review the [PyTorch optimization-loop pattern](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html); the new concept is direct embedding-space control.

### Current-state TODOs

- [ ] Make the access-model limitation prominent before implementation, not only after it.
- [ ] Ensure the exercise adds insight beyond repeating another generic Adam loop.
- [ ] Compare base versus prefixed behavior with an explicit score on held-out prompts and benign utility.
- [ ] Consider making this a shorter continuous-relaxation comparison or a stretch exercise.
