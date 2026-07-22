# 1.1 — LLM internals and chat serialization

## What to expect

Participants turn structured messages into the token stream an LLM actually
receives, then inspect how control tokens and vendor chat formats affect the
security boundary between instructions and data.

**Suggested time:** 75 minutes

**Exercises:** [Open the participant instructions](section1_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Python collections/JSON, basic HTTP API use, and the Hugging Face chat-template pre-reading | Make an OpenAI-compatible request, inspect a tokenizer chat template, and debug message serialization |
| ML | Tokens, next-token prediction, and basic sampling | Explain how roles, tool calls, and special tokens become model inputs |
| Security | Injection and parser-boundary intuition from conventional application security | Treat chat serialization and special-token handling as part of the model's security perimeter |
| Theory | Basic instruction-following model concepts | Distinguish a message hierarchy in an API from the flat token sequence processed by the model |

### Preparation

- Required prework: Hugging Face's [chat-template guide](https://huggingface.co/docs/transformers/chat_templating) and the relevant chapters of Karpathy's [*Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI).
- Just-in-time reference: the [Hugging Face chat-template documentation](https://huggingface.co/docs/transformers/chat_templating).

### Current-state TODOs

- [ ] Fix setup wording and make credential failures easy to diagnose.
- [ ] Briefly identify [Anthropic content blocks](https://docs.anthropic.com/en/api/messages) and [OpenAI Responses](https://developers.openai.com/api/reference/resources/responses) as formats
      participants will encounter, while keeping one API for the exercise.
- [ ] Qualify claims about production providers' handling of injected control tokens.
- [ ] Ensure tests assess the resulting serialized structure, not only string assembly.
