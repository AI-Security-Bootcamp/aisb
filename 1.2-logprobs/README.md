# 1.2 — Log probabilities

## What to expect

Participants use an LLM API's logprob response to connect next-token inference
with practical information leakage, extraction, and adversarial optimization.

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | The API workflow from 1.1 and basic Python object inspection | Request and parse token-level logprobs and handle top-k response structures |
| ML | Logarithms, softmax intuition, logits, and next-token prediction | Convert logprobs to relative probabilities and interpret ranks and uncertainty without treating them as calibrated confidence |
| Security | Basic API threat modeling | Explain how exposed distributions increase attacker information and support extraction or prompt optimization |
| Theory | Black-box versus white-box access | Relate API observability to attacker capability and defense trade-offs |

### Preparation

- Revisit the inference and sampling portions of Karpathy's LLM prework.
- Review 1.1 if OpenAI-compatible response objects are unfamiliar.

### Current-state TODOs

- [ ] Correct “completions endpoint” wording where the code uses chat completions.
- [ ] Turn the misuse question into a short think/pair/reveal discussion and link forward to GCG and distillation.
- [ ] Separate API-parsing competence from the more important interpretation and threat-analysis task.
- [ ] Discuss what top-k truncation hides and what it still leaks.
