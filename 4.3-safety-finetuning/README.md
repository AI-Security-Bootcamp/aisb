# 4.3 — Removing safety behavior with LoRA fine-tuning

## What to expect

Participants learn a PEFT/LoRA workflow and measure how a small adapter trained
on harmful completions changes refusal behavior and benign model utility.

**Suggested time:** 120 minutes

**Exercises:** [Open the participant instructions](section3_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | PyTorch training, Hugging Face model/tokenizer use, and basic dataset handling; PEFT, Trainer, and vLLM are new | Configure LoRA, inspect trainable parameters, train/save an adapter, and run comparative inference |
| ML | Supervised fine-tuning, held-out evaluation, and classification metrics | Explain low-rank adaptation and evaluate both intended behavioral change and regressions |
| Security | Model-artifact integrity, dual-use handling, and safety-evaluation basics | Demonstrate how small post-training access can weaken safeguards and define controls around adapters and deployment |
| Theory | Distinction between learned refusal behavior and underlying capability | Distinguish safety-removal fine-tuning from representation-direction ablation (“abliteration”) |

### Preparation

- Required prework: the [PyTorch Blitz](https://docs.pytorch.org/tutorials/beginner/blitz/index.html) and the post-training chapters of Karpathy's [*Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI).
- Just-in-time references: the official PEFT [LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora), Hugging Face's [fine-tuning guide](https://huggingface.co/docs/transformers/training), and [*LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B*](https://arxiv.org/abs/2310.20624).

### Current-state TODOs

- [x] Correct the participant-facing terminology: this is LoRA safety-removal fine-tuning, not representation-direction abliteration.
- [ ] Rename legacy function/artifact identifiers that still use `abliteration`, or add a separate genuine refusal-direction ablation exercise.
- [ ] Make PEFT/Trainer skills explicit and use progressive documentation links and checkpoints.
- [ ] Evaluate held-out harmful compliance and benign utility/regression, not refusal rate alone.
- [ ] Replace realistic dangerous output targets with safe proxies where possible and document adapter cleanup/handling.
- [ ] Reduce setup fragility from gated legacy models and several unfamiliar frameworks in one exercise.
