# 1.3 — Instruction hierarchies and assistant prefills

## What to expect

Participants experiment with assistant prefills and connect instruction priority
to the familiar cybersecurity idea of principals with different authority.

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Message construction and API calls from 1.1 | Construct and vary an assistant prefill and compare controlled outputs |
| ML | Autoregressive continuation and chat templates | Explain why a partial assistant message changes the continuation distribution |
| Security | Authorization principals, policy precedence, and injection | Map system/developer/user authority to principals and recognize prefill as a policy-bypass surface |
| Theory | Instruction-following and model conditioning | Distinguish desired instruction hierarchy from the statistical behavior that implements it |

### Preparation

- Complete 1.1 first.
- Revisit the prework on chat templates if role serialization is not yet comfortable.

### Current-state TODOs

- [ ] Provide the repeated API/message wrapper and make prefill design the central learner task.
- [ ] Add at least one playful, harmless prefill variation and ask learners to predict its effect.
- [ ] Draw the correspondence with authorization principals explicitly.
- [ ] Move the Day 1 summary into a standalone day-level summary so it is not hidden in an optional section.
