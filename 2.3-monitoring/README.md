# 2.3 — AI-control monitoring

## What to expect

Participants build and red-team a monitor for an untrusted coding policy, then
measure detection quality rather than judging a few anecdotes.

**Suggested time:** 90 minutes

**Exercises:** [Open the participant instructions](section3_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Python, structured logs, basic data analysis, and LLM API use | Run ControlArena evaluations, inspect trajectories, and turn monitor outputs into analyzable scores |
| ML | Classification scores and basic precision/recall intuition | Interpret score distributions, ROC curves, AUC, thresholds, and capability-gap effects |
| Security | Detection engineering, attacker adaptation, and false-positive trade-offs | Write a monitor policy, design an evasion policy, and evaluate both on held-out behavior |
| Theory | Introductory AI control and chain-of-thought concepts | Explain what monitoring assumes about trusted models, observability, and attacker knowledge |

### Preparation

- Recommended prework: BlueDot's [*Introduction to AI Control*](https://blog.bluedot.org/p/ai-control).
- Just-in-time references: the [ControlArena documentation](https://control-arena.aisi.org.uk/) and scikit-learn's [ROC/AUC guide](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics).

### Current-state TODOs

- [ ] State which ControlArena/API competencies are learning objectives and which are provided infrastructure.
- [ ] Use held-out attack families and report false positives at operationally meaningful base rates.
- [ ] Separate prompt iteration data from final evaluation data.
- [ ] Discuss adaptive pressure and limits before presenting chain-of-thought monitoring as an advantage.
