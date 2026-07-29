# 2.3 — AI-control monitoring

## What to expect

Participants build a monitor that flags suspicious actions taken by a coding
agent, test ways an attacker might evade it, and measure its performance across
many examples.

**Suggested time:** 90 minutes

**Exercises:** [Open the participant instructions](section3_instructions.md)

**Setup:** This section uses `control-arena`, which requires a newer `datasets`
than the main `requirements.txt` allows. Install it in a **separate virtual
environment** from `../requirements-control-arena.txt` (see that file's header for
the exact commands). Sections 2.3 and 2.4 share this environment.

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | - | Run ControlArena evaluations, inspect trajectories, and turn monitor outputs into analyzable scores |
| ML | Confusion matrices, precision, recall, and decision thresholds | Interpret score distributions, ROC curves, AUC, thresholds, and capability-gap effects |
| Security | Detector errors and adaptive attackers | Write a monitor policy, design an evasion policy, and evaluate both on held-out behavior |
| Theory | Trusted models, untrusted models, and monitors in AI control | Explain what monitoring assumes about trusted models, observability, and attacker knowledge |

### Background

- Read BlueDot's [*Introduction to AI Control*](https://blog.bluedot.org/p/ai-control).
- References: the [ControlArena documentation](https://control-arena.aisi.org.uk/) and scikit-learn's [ROC/AUC guide](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics).

### Current-state TODOs

- [ ] State which ControlArena/API competencies are learning objectives and which are provided infrastructure.
- [ ] Use held-out attack families and report false positives at operationally meaningful base rates.
- [ ] Separate prompt iteration data from final evaluation data.
- [ ] Discuss adaptive pressure and limits before presenting chain-of-thought monitoring as an advantage.
