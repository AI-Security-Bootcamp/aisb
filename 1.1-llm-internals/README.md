# 1.1 — LLM internals and chat serialization

## What to expect

Participants turn structured messages into the token stream an LLM actually
receives, then inspect how control tokens and vendor chat formats affect the
security boundary between instructions and data.

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Python collections/JSON, basic HTTP API use, and the Hugging Face chat-template pre-reading | Make an OpenAI-compatible request, inspect a tokenizer chat template, and debug message serialization |
| ML | Tokens, next-token prediction, and basic sampling | Explain how roles, tool calls, and special tokens become model inputs |
| Security | Injection and parser-boundary intuition from conventional application security | Treat chat serialization and special-token handling as part of the model's security perimeter |
| Theory | Basic instruction-following model concepts | Distinguish a message hierarchy in an API from the flat token sequence processed by the model |

### Preparation

- Required prework: *Chat Templates on Hugging Face* and the LLM chapters of the AISB reading list.
- Just-in-time reference: Hugging Face chat-template documentation.

### Current-state TODOs

- [ ] Fix setup wording and make credential failures easy to diagnose.
- [ ] Briefly identify Anthropic content blocks and OpenAI Responses as formats
      participants will encounter, while keeping one API for the exercise.
- [ ] Qualify claims about production providers' handling of injected control tokens.
- [ ] Ensure tests assess the resulting serialized structure, not only string assembly.
