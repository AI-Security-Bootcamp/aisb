# Day 1 pre-reading: how LLMs are trained

## What to expect

This reading should supply a compact mental model of the LLM lifecycle so the
labs can focus on security-relevant interfaces rather than reteaching the whole
training pipeline.

**Suggested time:** 30 minutes for this section, in addition to the required prework

**Reading:** [Open the LLM training guide](llm-training-guide.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | General software and systems experience | Recognize the major data, compute, training, evaluation, and serving components of an LLM stack |
| ML | Introductory neural networks and gradient descent from the required prework | Explain tokenization, embeddings, attention, next-token training, supervised fine-tuning, and preference optimization at a working level |
| Security | Experience reasoning about assets and trust boundaries | Identify where data, weights, evaluations, APIs, and infrastructure create security boundaries |
| Theory | High-level familiarity with AI alignment | Distinguish capability training, instruction tuning, and preference/safety training and explain why each stage matters for safety |

### Preparation

- Required prework: 3Blue1Brown's [*But what is a neural network?*](https://www.3blue1brown.com/lessons/neural-networks) and [*Gradient descent, how neural networks learn*](https://www.3blue1brown.com/lessons/gradient-descent), plus Andrej Karpathy's [*Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI) through 2:27:47.
- Use [llm-training-guide.md](llm-training-guide.md) as the section reading.
- If PPO is unfamiliar, revisit Ang Li-Lian's [*A simple technical explanation of RLH(AI)F*](https://kairos.fm/simple-technical-rlhaif/) before the preference-training questions.

### Current-state TODOs

- [x] Correct the [OLMo paper](https://arxiv.org/abs/2501.00656) title everywhere to *2 OLMo 2 Furious*.
- [x] Teach token → embedding → position → attention before asking about RoPE.
- [x] Introduce PPO and classic RLHF before asking learners to compare them with DPO.
- [ ] Replace compressed, slogan-like claims with patient explanations and mark toy intuitions as such.
- [ ] Re-estimate the advertised reading time against the actual videos and paper sections.
