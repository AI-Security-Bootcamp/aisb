# Issue 39 fix validation — 2026-09-16

Branch: `fix/39-resumable-monitor-evaluation`.
PR base: `fix/38-monitor-request-logging` (the logging fix).

## Full real-model run

Ran the actual course `create_monitor` and `evaluate_monitor_on_trajectories`
functions against all 50 honest and 50 attack APPS trajectories from the original
sandbox reproduction. That replay passed all 100 main tasks and 50 attack side
tasks. The fixed monitor used the reference prompt and the same OpenRouter Llama
3.1 8B model. It did not regenerate trajectories or remove samples.

| Pass | New monitor evaluations | Valid honest scores | Valid attack scores | Elapsed from start |
| --- | ---: | ---: | ---: | ---: |
| 1 | 100 | 44/50 | 45/50 | 314.65 s |
| 2 | 11 | 49/50 | 49/50 | 380.90 s |
| 3 | 2 | 50/50 | 49/50 | 405.19 s |
| 4 | 1 | 50/50 | 50/50 | 409.84 s |

The first pass encountered six request timeouts and five malformed scores. The
second pass recorded an overall sample deadline and a malformed score. The third
pass had one malformed score. The fourth completed the remaining sample. Each
rerun preserved completed samples and changed only the failed samples' monitor
events (100 → 11 → 2 → 1 evaluations, verified from their recorded completion
times). Failed logs remain available for inspection.

No AUC was returned for the first three passes. At full coverage, the helper
produced **AUC 0.5994**, and the ROC plot was saved. This number documents this
stochastic run; it is not a performance guarantee for the monitor.

While the first pass was still running, reading the native `.eval` files showed
49 saved sample outcomes, including five errors. Completed work was already on
disk before the batch finished. After full completion, a fresh Python process
reused the completed evals. The verification checks stable eval IDs, unchanged
eval-file sets, and identical score arrays/AUC. An initial diagnostic assertion
compared plain paths with `file://` URIs; it was corrected to compare eval identity
and files. The final verification passed without new monitor evaluations.

## Behavior and scope

- At most four samples run concurrently. Model calls have a 45-second attempt
  deadline, zero Inspect retries, and a 2,048-token output cap; each entire saved
  trajectory has a 60-second deadline, including queueing and all monitor steps.
- Failures are isolated per sample and saved with the request/error transcript
  supplied by the issue #38 fix. A distinct monitor-timeout exception makes an
  expired deadline retryable; Inspect treats a plain timeout as a sample limit.
- Missing, malformed, nonfinite, or out-of-range scores cannot become numeric
  safe observations. Old message-level monitor scores are cleared before running
  a new monitor. Course monitor response caching is disabled so malformed outputs
  receive a fresh attempt; native evaluation checkpoints reuse successful work.
- `IncompleteMonitorEvaluation` reports honest/attack coverage and the log path.
  The caller can rerun the same cell. AUC calculation requires complete valid
  coverage; the helper does not silently plot a selected subset.
- Inspect's eval-set identity tracks monitor configuration. A dataset-content
  hash also invalidates reuse when source trajectories or metadata change.

Long valid explanations may hit the output cap; a resulting malformed response
is reported and retried rather than silently counted as safe. The fix bounds and
recovers from failures; it cannot guarantee a model provider will return a valid
score on a particular attempt.

## Regression validation

**10 tests passed locally (Python 3.11.2) and on the dev box (Python 3.12.14).**
They cover the inherited logging/cache/cancellation checks plus:

- a real 100-sample Inspect evaluation with one controlled stalled response;
  99 scores are retained, and rerunning calls only the failed sample;
- provider errors and malformed scores, including stale-score removal;
- completed-run resumption after a Python restart without model calls;
- prompt and source-trajectory changes invalidating prior scores;
- the actual solution helper refusing to calculate AUC after an incomplete run.

The original 99% stall and the timeout-only batch failure were also reproduced
again from commit `3bc387e` using `reproduce.py` and `reproduce.py --bounded`.
The instruction build passed, and generated Markdown/tests were rebuilt from
`section2_solution.py`.

```bash
python -m pytest 2.2-monitoring/tests -q
./build-instructions.sh 2.2-monitoring/section2_solution.py
python reproductions/issue_39/reproduce.py
python reproductions/issue_39/reproduce.py --bounded \
  --output reproductions/issue_39/artifacts/baseline-bounded
python reproductions/issue_39/verify_fix.py \
  --input /path/to/original/apps-replay-100 \
  --output reproductions/issue_39/artifacts/fixed-full-100 \
  --env-file /path/to/credentials.env
```

Real-run artifacts are in `artifacts/fixed-full-100/` locally and under
`/root/aisb-fix39/reproductions/issue_39/artifacts/` remotely. `initial-passes.json`
contains all four fresh passes; `summary.json` records the successful completed-run
reuse check; `scored/` retains native evals; `roc.png` is the complete ROC.
Dependencies remain ControlArena 17.1.1, Inspect 0.3.244, and OpenAI client 2.54.0.
