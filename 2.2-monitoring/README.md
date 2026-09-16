# 2.2 — AI-control monitoring

## What to expect

Participants build a monitor that flags suspicious actions taken by a coding
agent, test ways an attacker might evade it, and measure its performance across
many examples.

**Suggested time:** 90 minutes

**Exercises:** [Open the participant instructions](section2_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | - | Run ControlArena evaluations; locate a monitor's request, response, or interruption in an Inspect Transcript; turn monitor outputs into scores |
| ML | Given a confusion matrix, calculate precision and recall; explain how a decision threshold changes false positives and false negatives | Interpret score distributions, ROC curves, AUC, thresholds, and capability-gap effects |
| Security | - | Write a monitor policy, design an evasion policy, and evaluate both on held-out behavior |
| Theory | Identify the trusted monitor and untrusted agent in a control protocol and state what each can observe | Explain what monitoring assumes about trusted models, observability, and attacker knowledge |

### Background

- Read BlueDot's [*Introduction to AI Control*](https://blog.bluedot.org/p/ai-control).
- References: the [ControlArena documentation](https://control-arena.aisi.org.uk/) and scikit-learn's [ROC/AUC guide](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics).
- Read Inspect's concise [log-viewer guide](https://inspect.aisi.org.uk/log-viewer.html)
  before inspecting monitor requests. Saved-trajectory monitoring writes separate
  evals under `logs/*_scored/`; open their **Transcript** tab. Cancelled calls show
  the request and an interruption error, without a completed response.

### Incomplete monitoring runs

The evaluator limits concurrent samples and bounds request and sample time.
It saves completed samples and reports coverage for honest and attack runs.
Timeouts, provider errors, and malformed scores stop ROC/AUC calculation with
a message pointing to the saved logs. Rerun the same evaluation to retry only
failed or unfinished samples. Changing the prompt or trajectories starts a
separate evaluation; old scores are not mixed into the new result.
