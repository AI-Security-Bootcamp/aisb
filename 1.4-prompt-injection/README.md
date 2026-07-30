# 1.4 — Prompt injection and RAG poisoning

## What to expect

Participants threat-model an LLM application, probe a defended RAG system, and
craft an indirect injection against the boundary between instructions and data.

**Suggested time:** 75–90 minutes

**Exercises:** [Open the participant instructions](section4_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | RAG and MCP architecture | Operate a small RAG application, modify its knowledge base, and run a controlled attack |
| ML | Context windows and prompt serialization | Explain why retrieved text and trusted instructions share a model context despite application-level delimiters |
| Security | Sources, sinks, trust boundaries, and attacker-controlled input | Enumerate untrusted channels, perform black-box reconnaissance, and distinguish mitigations from complete prevention |
| Theory | - | Explain why prompt injection is partly an architectural control problem rather than only a prompt-writing problem |

### Background

- Required prework: Google's [RAG](https://cloud.google.com/use-cases/retrieval-augmented-generation) and [MCP](https://cloud.google.com/discover/what-is-model-context-protocol) explainers, [MITRE ATLAS / *AI Security 101*](https://atlas.mitre.org/), and Karpathy's [*Intro to Large Language Models*](https://www.youtube.com/watch?v=zjkBMFhNj_g) security section.

### Current-state TODOs

- [ ] Move the defended system prompt and document formatting behind an opaque
      helper; they are currently visible despite the reconnaissance framing.
- [ ] Evaluate retrieval success separately from instruction-following success.
- [ ] Replace one exact live-model output check with repeated or fixture-backed evaluation.
- [ ] Ask for an explicit mitigation analysis after successful exploitation.
