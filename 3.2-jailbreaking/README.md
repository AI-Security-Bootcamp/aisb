# 3.2 — Jailbreaking

## What to expect

Participants conduct a short, bounded jailbreak exercise so later guardrail work
is grounded in direct experience of attacker adaptation.

**Suggested time:** 20 minutes

**Exercises:** [Open the participant instructions](section2_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Ability to operate the selected hosted or local challenge | Record prompts, outcomes, and enough context to reproduce an observation |
| ML | Statistical instruction following and safety post-training | Explain why jailbreak success varies across models, prompts, and sampling conditions |
| Security | Authorized testing, red-team discipline, and prompt-injection concepts | Generate and categorize jailbreak strategies without confusing one success with general effectiveness |
| Theory | Basic alignment and dangerous-capability evaluation concepts | Relate empirical jailbreaks to the limits of behavioral safety training |

### Preparation

- Required prework: Karpathy's [*Intro to Large Language Models*](https://www.youtube.com/watch?v=zjkBMFhNj_g) security section and Google DeepMind's [dangerous-capability evaluations overview](https://deepmind.google/research/publications/78150/).
- Follow instructor guidance on authorized services and harmless test targets.

### Current-state TODOs

- [ ] Provide a local harmless fallback if external challenge services are unavailable or unsuitable.
- [ ] Define the required artifact: attempts, observed outcomes, and 2–3 categorized strategies.
- [ ] State service-use, data-retention, and responsible-testing expectations.
- [ ] Run this synchronously and enforce the timebox so it does not consume the guardrail sections.
