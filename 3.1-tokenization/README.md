# 3.1 — Tokenization and prompt construction

## What to expect

Participants move from hosted APIs to local models, using tokenizers and chat
templates directly to investigate prompt construction and model-specific
reasoning controls.

**Suggested time:** 45 minutes

**Exercises:** [Open the participant instructions](section1_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Python, remote-machine access established on Day 0, and Hugging Face chat-template prework | Load a local tokenizer/model, apply templates, inspect token IDs, and run generation |
| ML | Tokens, autoregressive inference, sampling, and basic model-family differences | Explain generation prompts, continuation of an existing assistant message, and model-specific reasoning modes |
| Security | Prompt-boundary and control-token lessons from Day 1 | Recognize prompt construction as an attack surface and identify assumptions made by downstream guardrails |
| Theory | High-level chain-of-thought concepts | Distinguish observed output differences from causal claims about reasoning or safety |

### Preparation

- Required prework: Hugging Face [chat templates](https://huggingface.co/docs/transformers/chat_templating) and the inference chapters of Karpathy's [*Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI).
- Refresh the [Hugging Face generation API](https://huggingface.co/docs/transformers/main_classes/text_generation) as needed; remote GPU access should already have been validated.

### Current-state TODOs

- [ ] Move remote SSH and model-cache setup to Day 0.
- [ ] Phrase `continue_final_message` behavior empirically rather than claiming it inherently creates an infinite loop.
- [ ] Use a controlled model/configuration comparison when discussing thinking versus non-thinking behavior.
- [ ] Test prompt construction and interpretation, not the incidental presence of a literal `<think>` tag.
