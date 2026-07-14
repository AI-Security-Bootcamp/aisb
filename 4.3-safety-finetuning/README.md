# 4.3 — Removing safety behavior with LoRA fine-tuning

## What to expect

Participants learn a PEFT/LoRA workflow and measure how a small adapter trained
on harmful completions changes refusal behavior and benign model utility.

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | PyTorch training, Hugging Face model/tokenizer use, and basic dataset handling; PEFT, Trainer, and vLLM are new | Configure LoRA, inspect trainable parameters, train/save an adapter, and run comparative inference |
| ML | Supervised fine-tuning, held-out evaluation, and classification metrics | Explain low-rank adaptation and evaluate both intended behavioral change and regressions |
| Security | Model-artifact integrity, dual-use handling, and safety-evaluation basics | Demonstrate how small post-training access can weaken safeguards and define controls around adapters and deployment |
| Theory | Distinction between learned refusal behavior and underlying capability | Distinguish safety-removal fine-tuning from representation-direction ablation (“abliteration”) |

### Preparation

- Required prework: PyTorch Blitz and the LLM post-training chapters.
- Just-in-time references: official PEFT LoRA and Hugging Face Trainer examples; the assigned LoRA safety-removal paper.

### Current-state TODOs

- [x] Correct the participant-facing terminology: this is LoRA safety-removal fine-tuning, not representation-direction abliteration.
- [ ] Rename legacy function/artifact identifiers that still use `abliteration`, or add a separate genuine refusal-direction ablation exercise.
- [ ] Make PEFT/Trainer skills explicit and use progressive documentation links and checkpoints.
- [ ] Evaluate held-out harmful compliance and benign utility/regression, not refusal rate alone.
- [ ] Replace realistic dangerous output targets with safe proxies where possible and document adapter cleanup/handling.
- [ ] Reduce setup fragility from gated legacy models and several unfamiliar frameworks in one exercise.
