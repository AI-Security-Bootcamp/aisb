# 3.3 — Guardrails: attacks and defenses

## What to expect

Participants build progressively stronger input/output defenses, attack each
layer, and compare lexical, model-based, and representation-based detection.

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Local generation from 3.1, API/model invocation, and basic sklearn-style evaluation | Implement a generation wrapper, classifier pipeline, activation extraction, and simple probe evaluation |
| ML | Classification metrics, hidden representations, train/test separation, and basic linear models | Explain what each detector observes and evaluate generalization on held-out attack and benign families |
| Security | Jailbreak experience from 3.2 and detection-engineering concepts | Bypass brittle filters, design layered controls, and reason about adaptive attackers and false positives |
| Theory | Behavioral versus representation-level safety signals | Distinguish evidence about one dataset from claims about semantic intent or robust safety |

### Preparation

- Complete 3.1–3.2.
- Just-in-time references: deterministic generation controls, precision/recall, and sklearn pipelines/cross-validation.

### Current-state TODOs

- [ ] Replace visible blocklist “reverse engineering” with probing of an opaque filter or a short attack-design discussion.
- [ ] Let participants write the classifier policy while making the intended inference/API work explicit.
- [ ] Hide the novel-writing answer until after a genuine, timeboxed attempt.
- [ ] Rebuild the probe experiment with a leakage-free pipeline, substantially more data, held-out attack families, lexical baselines, and FPR/TPR reporting.
- [ ] Remove claims that one probe detects semantic intent or uses representations an input attacker cannot manipulate.
- [ ] Mark the probe as stretch or provide precomputed activations if runtime threatens the core route.
