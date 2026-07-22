# 5.1 — Adversarial attacks on vision models

## What to expect

Participants apply autograd to optimize an input rather than model weights,
building targeted attacks under perturbation constraints and measuring the
success/visibility trade-off.

**Suggested time:** 60 minutes

**Exercises:** [Open the participant instructions](section1_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | PyTorch tensors and autograd | Load a classifier, differentiate with respect to input pixels, and implement iterative constrained optimization |
| ML | Classification logits and loss, L2/L∞ norms, input normalization, and eval mode | Explain targeted versus untargeted objectives and evaluate attack success across perturbation budgets |
| Security | Adversarial-example threat models | State attacker knowledge/control, avoid overgeneralizing from one image, and identify deployment transformations that may break an attack |
| Theory | - | Explain why small norm-bounded changes can cross a model decision boundary without calling logits calibrated confidence |

### Background

- Required prework: the [PyTorch Blitz](https://docs.pytorch.org/tutorials/beginner/blitz/index.html), [*But what is a neural network?*](https://www.3blue1brown.com/lessons/neural-networks), and [*Gradient descent, how neural networks learn*](https://www.3blue1brown.com/lessons/gradient-descent).
- Just-in-time references: the official PyTorch [adversarial-example tutorial](https://docs.pytorch.org/tutorials/beginner/fgsm_tutorial.html) and [autograd guide](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html).

### Current-state TODOs

- [x] Correct the “untargeted” description: the current objective specifies a target class.
- [ ] Apply and constrain perturbations in pixel space, or derive valid bounds in normalized model space; do not clamp normalized ViT tensors to `[0, 1]`.
- [ ] Evaluate several images/targets and plot success against epsilon rather than relying on one anecdote.
- [ ] Add transfer or transformation robustness and make the attacker-access model explicit.
