# 1.2 — Log probabilities

## What to expect

Participants use an LLM API's logprob response to connect next-token inference
with practical information leakage, extraction, and adversarial optimization.

**Suggested time:** 45 minutes

**Exercises:** [Open the participant instructions](section2_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | - | Request and parse token-level logprobs and handle top-k response structures |
| ML | Logarithms, softmax, and next-token probability distributions | Convert logprobs to relative probabilities and interpret ranks and uncertainty without treating them as calibrated confidence |
| Security | Black-box API access models | Explain how exposed distributions increase attacker information and support extraction or prompt optimization |
| Theory | Model access levels: outputs, weights, gradients, and activations | Relate API observability to attacker capability and defense trade-offs |

### Background

- Revisit the inference and sampling portions of Karpathy's [*Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI).
