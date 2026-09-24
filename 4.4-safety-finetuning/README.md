# 4.4 — Removing safety behavior with LoRA fine-tuning (Optional)

> **Optional:** Section 4.3 is the current exercise and directly
> studies representation-direction abliteration. This alternative LoRA exercise is
> retained for comparison and should only be completed if an instructor
> recommends it.

## What to expect

Participants learn a PEFT/LoRA workflow and measure how a small adapter trained
on harmful completions changes refusal behavior and benign model utility.

**Suggested time:** 120 minutes

**Exercises:** [Open the participant instructions](section4_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | PyTorch training loops | Configure LoRA, inspect trainable parameters, train/save an adapter, and run comparative inference |
| ML | Supervised fine-tuning and held-out evaluation | Explain low-rank adaptation and evaluate both intended behavioral change and regressions |
| Security | Model/adapter integrity and safe handling of dual-use outputs | Demonstrate how small post-training access can weaken safeguards and define controls around adapters and deployment |
| Theory | - | Distinguish safety-removal fine-tuning from representation-direction ablation (“abliteration”) |

### Background

- Required prework: the [PyTorch Blitz](https://docs.pytorch.org/tutorials/beginner/blitz/index.html) and Karpathy's [*Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI) through 2:27:47, focusing on the post-training chapters.
- Review the official PEFT [LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora) and Hugging Face [fine-tuning guide](https://huggingface.co/docs/transformers/training). Read the abstract, method, and Figure 1 of [*LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B*](https://arxiv.org/abs/2310.20624); the full paper is optional longer reading.
- References: Hugging Face Datasets [transformations](https://huggingface.co/docs/datasets/process) and vLLM's [LoRA adapter guide](https://docs.vllm.ai/en/stable/features/lora/).
