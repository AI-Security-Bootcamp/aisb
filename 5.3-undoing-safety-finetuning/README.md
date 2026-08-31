# 5.3 — Undoing safety fine-tuning: refusal direction and abliteration

## What to expect

Participants recover a refusal direction from model activations, test whether
that direction is necessary and sufficient for refusal, bake the intervention
into the model's weights, and measure the effect on general capability.

**Suggested time:** 150 minutes

**Exercises:** [Open the participant instructions](section3_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Tensor manipulation in PyTorch | Capture and modify residual-stream activations with hooks, edit model weights, and compare two model variants with Inspect |
| ML | - | Recover a behavioral direction with difference-in-means and test necessity, near-sufficiency, and capability regressions |
| Security | - | Explain why open-weight refusal training can be removed cheaply and what this implies for relying on model-level safeguards |
| Theory | Vectors, dot products, means, and matrix multiplication | Implement vector rejection and derive the equivalent weight-orthogonalization update |

### Background

- Required prework: the [PyTorch Blitz](https://docs.pytorch.org/tutorials/beginner/blitz/index.html) and Karpathy's [*Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI) through 2:27:47.
- Read the abstract, introduction, and Figure 1 of Arditi et al., [*Refusal in Language Models Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717).
- References: PyTorch [`register_forward_pre_hook`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook), Hugging Face [chat templates](https://huggingface.co/docs/transformers/chat_templating), and the Inspect [evaluation tutorial](https://inspect.aisi.org.uk/tutorial.html).
