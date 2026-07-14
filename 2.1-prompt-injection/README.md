# 2.1 — Prompt injection and RAG poisoning

## What to expect

Participants threat-model an LLM application, probe a defended RAG system, and
craft an indirect injection against the boundary between instructions and data.

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Basic LLM API use plus high-level RAG and MCP prework | Operate a small RAG application, modify its knowledge base, and run a controlled attack |
| ML | Context windows and instruction following | Explain why retrieved text and trusted instructions share a model context despite application-level delimiters |
| Security | Injection attacks, trust boundaries, reconnaissance, and defense in depth | Enumerate untrusted channels, perform black-box reconnaissance, and distinguish mitigations from complete prevention |
| Theory | Basic LLM application architecture | Explain why prompt injection is partly an architectural control problem rather than only a prompt-writing problem |

### Preparation

- Required prework: Google's RAG and MCP explainers, MITRE *AI Security 101*, and Karpathy's LLM Security introduction.
- Review 1.1–1.3 for message hierarchy and serialization.

### Current-state TODOs

- [ ] Move the defended system prompt and document formatting behind an opaque
      helper; they are currently visible despite the reconnaissance framing.
- [ ] Evaluate retrieval success separately from instruction-following success.
- [ ] Replace one exact live-model output check with repeated or fixture-backed evaluation.
- [ ] Ask for an explicit mitigation analysis after successful exploitation.
