# Day 1 pre-reading: how LLMs are trained

## What to expect

This reading should supply a compact mental model of the LLM lifecycle so the
labs can focus on security-relevant interfaces rather than reteaching the whole
training pipeline.

**Suggested time:** 30 minutes for this section, in addition to the required prework

**Reading:** [Open the LLM training guide](llm-training-guide.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | - | Draw and label the data, compute, training, evaluation, and serving components of an LLM stack |
| ML | Neural-network fundamentals and gradient descent | Explain tokenization, embeddings, attention, next-token training, supervised fine-tuning, and preference optimization, and place them in training order |
| Security | Assets, trust boundaries, and attacker objectives | Mark where training data, weights, evaluations, APIs, and infrastructure introduce assets and trust boundaries in an LLM lifecycle |
| Theory | AI alignment and alignment failures | Distinguish capability training, instruction tuning, and preference/safety training and explain the safety purpose of each stage |

### Background

- Required prework: 3Blue1Brown's [*But what is a neural network?*](https://www.3blue1brown.com/lessons/neural-networks) and [*Gradient descent, how neural networks learn*](https://www.3blue1brown.com/lessons/gradient-descent), plus Andrej Karpathy's [*Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI) through 2:27:47.
- Use [llm-training-guide.md](llm-training-guide.md) as the section reading.
- Read Ang Li-Lian's [*A simple technical explanation of RLH(AI)F*](https://anglilian.com/blog/a-simple-technical-explanation-of-rlhaif) before the preference-training questions, including the explanation of PPO-style RLHF.

### Current-state TODOs

- [x] Correct the [OLMo paper](https://arxiv.org/abs/2501.00656) title everywhere to *2 OLMo 2 Furious*.
- [x] Teach token → embedding → position → attention before asking about RoPE.
- [x] Introduce PPO and classic RLHF before asking learners to compare them with DPO.
- [ ] Replace compressed, slogan-like claims with patient explanations and mark toy intuitions as such.
- [ ] Re-estimate the advertised reading time against the actual videos and paper sections.
