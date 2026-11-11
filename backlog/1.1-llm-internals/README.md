# 1.1 — LLM internals and chat serialization

## What to expect

Participants turn structured messages into the token stream an LLM actually
receives, then inspect how control tokens and vendor chat formats affect the
security boundary between instructions and data.

**Suggested time:** 75 minutes

**Exercises:** [Open the participant instructions](section1_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | - | Make an OpenAI-compatible request, inspect a tokenizer chat template, and debug the resulting message serialization |
| ML | Tokenization, next-token prediction, and sampling temperature | Explain how roles, tool calls, and special tokens become one model input sequence |
| Security | Untrusted input and parser boundaries | Identify the role, special-token, and serialization boundaries that form the model API's security perimeter |
| Theory | - | Distinguish the message hierarchy represented by an API from the flat token sequence processed by the model |

### Background

- Required prework: Hugging Face's [chat-template guide](https://huggingface.co/docs/transformers/chat_templating) and Karpathy's [*Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI) through 2:27:47.
