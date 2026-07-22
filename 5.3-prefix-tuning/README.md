# 5.3 — Continuous adversarial prefixes

## What to expect

Participants optimize a shared prefix directly in embedding space, compare that
continuous relaxation with GCG, and examine why a strong optimization result may
not constitute a deployable API attack.

**Suggested time:** 30 minutes

**Exercises:** [Open the participant instructions](section3_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | PyTorch optimization | Construct `inputs_embeds`, optimize a shared latent tensor, and evaluate it on held-out prompts |
| ML | Batch losses, embedding vectors, and token IDs | Explain why latent optimization is easier than discrete token search and measure transfer across examples |
| Security | Text, token-ID, and embedding-input access models | Identify the narrow settings in which an attacker can supply embeddings and avoid presenting the result as an ordinary API jailbreak |
| Theory | Continuous relaxations and upper-bound experiments | Use latent-prefix success as evidence about optimization/capability, not automatically about real-world exploitability |

### Background

- Complete [5.2](../5.2-adversarial-language/README.md) first.
- Review the [PyTorch optimization-loop pattern](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html); the new concept is direct embedding-space control.

### Current-state TODOs

- [ ] Make the access-model limitation prominent before implementation, not only after it.
- [ ] Ensure the exercise adds insight beyond repeating another generic Adam loop.
- [ ] Compare base versus prefixed behavior with an explicit score on held-out prompts and benign utility.
- [ ] Consider making this a shorter continuous-relaxation comparison or a stretch exercise.
