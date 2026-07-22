# 2.3 — AI-control monitoring

## What to expect

Participants build and red-team a monitor for an untrusted coding policy, then
measure detection quality rather than judging a few anecdotes.

**Suggested time:** 90 minutes

**Exercises:** [Open the participant instructions](section3_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | - | Run ControlArena evaluations, inspect trajectories, and turn monitor outputs into analyzable scores |
| ML | Given a confusion matrix, calculate precision and recall and predict how moving a score threshold changes them | Interpret score distributions, ROC curves, AUC, thresholds, and capability-gap effects |
| Security | Given a detector, state the impact of a false positive and false negative and one way an attacker could adapt | Write a monitor policy, design an evasion policy, and evaluate both on held-out behavior |
| Theory | Define the trusted model, untrusted model, and monitor in an AI-control setup | Explain what monitoring assumes about trusted models, observability, and attacker knowledge |

### Background

- Read BlueDot's [*Introduction to AI Control*](https://blog.bluedot.org/p/ai-control).
- Just-in-time references: the [ControlArena documentation](https://control-arena.aisi.org.uk/) and scikit-learn's [ROC/AUC guide](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics).

### Current-state TODOs

- [ ] State which ControlArena/API competencies are learning objectives and which are provided infrastructure.
- [ ] Use held-out attack families and report false positives at operationally meaningful base rates.
- [ ] Separate prompt iteration data from final evaluation data.
- [ ] Discuss adaptive pressure and limits before presenting chain-of-thought monitoring as an advantage.
