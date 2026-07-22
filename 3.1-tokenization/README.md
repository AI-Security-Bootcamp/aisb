# 3.1 — Tokenization and prompt construction

## What to expect

Participants move from hosted APIs to local models, using tokenizers and chat
templates directly to investigate prompt construction and model-specific
reasoning controls.

**Suggested time:** 45 minutes

**Exercises:** [Open the participant instructions](section1_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | - | Load a local tokenizer/model, apply templates, inspect token IDs, and run generation |
| ML | Tokenization, autoregressive generation, and sampling temperature | Explain generation prompts, continuation of an existing assistant message, and model-specific reasoning modes |
| Security | Instruction, user-data, and control-token boundaries | Identify prompt-construction attack surfaces and list the assumptions made by downstream guardrails |
| Theory | Chain-of-thought generation and faithfulness limitations | Distinguish observed output differences from causal claims about reasoning or safety |

### Background

- Required prework: Hugging Face [chat templates](https://huggingface.co/docs/transformers/chat_templating) and the inference chapters of Karpathy's [*Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI).
- Review the [Hugging Face generation API](https://huggingface.co/docs/transformers/main_classes/text_generation); remote GPU access should already have been validated.

### Current-state TODOs

- [ ] Move remote SSH and model-cache setup to Day 0.
- [ ] Phrase `continue_final_message` behavior empirically rather than claiming it inherently creates an infinite loop.
- [ ] Use a controlled model/configuration comparison when discussing thinking versus non-thinking behavior.
- [ ] Test prompt construction and interpretation, not the incidental presence of a literal `<think>` tag.
