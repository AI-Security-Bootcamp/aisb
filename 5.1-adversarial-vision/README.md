# 5.1 — Adversarial attacks on vision models

## What to expect

Participants apply autograd to optimize an input rather than model weights,
building targeted attacks under perturbation constraints and measuring the
success/visibility trade-off.

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | PyTorch tensors/autograd, optimizer loops, and basic image-array handling | Load a classifier, differentiate with respect to input pixels, and implement iterative constrained optimization |
| ML | Classification logits/loss, normalization, L2/L∞ norms, and train-versus-eval mode | Explain targeted versus untargeted objectives and evaluate attack success across perturbation budgets |
| Security | Threat modeling and robustness testing | State attacker knowledge/control, avoid overgeneralizing from one image, and identify deployment transformations that may break an attack |
| Theory | Decision boundaries and high-dimensional inputs | Explain why small norm-bounded changes can cross a model decision boundary without calling logits calibrated confidence |

### Preparation

- Required prework: PyTorch Blitz and the neural-network/gradient-descent videos.
- Just-in-time reference: the official PyTorch adversarial-example tutorial and autograd documentation.

### Current-state TODOs

- [x] Correct the “untargeted” description: the current objective specifies a target class.
- [ ] Apply and constrain perturbations in pixel space, or derive valid bounds in normalized model space; do not clamp normalized ViT tensors to `[0, 1]`.
- [ ] Evaluate several images/targets and plot success against epsilon rather than relying on one anecdote.
- [ ] Add transfer or transformation robustness and make the attacker-access model explicit.
