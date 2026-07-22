# 1.3 — Instruction hierarchies and assistant prefills

## What to expect

Participants experiment with assistant prefills and connect instruction priority
to the familiar cybersecurity idea of principals with different authority.

**Suggested time:** 30 minutes (optional)

**Exercises:** [Open the participant instructions](section3_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Construct a multi-role message list and make the API call from 1.1 | Construct and vary an assistant prefill and compare controlled outputs |
| ML | Explain autoregressive continuation and apply a chat template to a message sequence | Explain why a partial assistant message changes the continuation distribution |
| Security | Given conflicting instructions from two principals, identify which principal has authority and which policy should win | Map system/developer/user authority to principals and identify prefill as a policy-bypass surface |
| Theory | - | Distinguish a desired instruction hierarchy from the statistical behavior used to implement it |

### Background

- Complete [1.1](../1.1-llm-internals/README.md) first.
- Revisit the [Hugging Face chat-template guide](https://huggingface.co/docs/transformers/chat_templating) if role serialization is not yet comfortable.

### Current-state TODOs

- [ ] Provide the repeated API/message wrapper and make prefill design the central learner task.
- [ ] Add at least one playful, harmless prefill variation and ask learners to predict its effect.
- [ ] Draw the correspondence with authorization principals explicitly.
- [ ] Move the Day 1 summary into a standalone day-level summary so it is not hidden in an optional section.
