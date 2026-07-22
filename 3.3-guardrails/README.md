# 3.3 — Guardrails: attacks and defenses

## What to expect

Participants build progressively stronger input/output defenses, attack each
layer, and compare lexical, model-based, and representation-based detection.

**Suggested time:** 90 minutes for the core route; up to 45 additional minutes for the probe stretch work

**Exercises:** [Open the participant instructions](section3_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Run local generation from 3.1, fit a scikit-learn classifier from `X`/`y`, and create a train/test split | Implement a generation wrapper, classifier pipeline, activation extraction, and simple probe evaluation |
| ML | Given a confusion matrix, calculate precision and recall; distinguish training from test data; identify a hidden-state vector | Explain what each detector observes and evaluate generalization on held-out attack and benign families |
| Security | Bypass a simple denylist and state how an adaptive attacker and false positives affect a detector | Bypass brittle filters, design layered controls, and reason about adaptive attackers and false positives |
| Theory | Distinguish a behavioral signal in text from a representation-level signal in activations | Distinguish evidence about one dataset from claims about semantic intent or robust safety |

### Background

- Complete [3.1](../3.1-tokenization/README.md) and [3.2](../3.2-jailbreaking/README.md).
- Just-in-time references: Hugging Face [generation controls](https://huggingface.co/docs/transformers/main_classes/text_generation), scikit-learn's [precision/recall guide](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics), [pipelines](https://scikit-learn.org/stable/modules/compose.html#pipeline), and [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).

### Current-state TODOs

- [ ] Replace visible blocklist “reverse engineering” with probing of an opaque filter or a short attack-design discussion.
- [ ] Let participants write the classifier policy while making the intended inference/API work explicit.
- [ ] Hide the novel-writing answer until after a genuine, timeboxed attempt.
- [ ] Rebuild the probe experiment with a leakage-free pipeline, substantially more data, held-out attack families, lexical baselines, and FPR/TPR reporting.
- [ ] Remove claims that one probe detects semantic intent or uses representations an input attacker cannot manipulate.
- [ ] Mark the probe as stretch or provide precomputed activations if runtime threatens the core route.
