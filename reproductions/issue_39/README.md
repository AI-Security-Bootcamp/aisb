# Issue 39: ROC progress stalls near completion

The progress bar belongs to post-hoc monitor calls. The course waits for every
call before calculating AUC, saving scored logs, or plotting. One unfinished
monitor can therefore hold the bar at 99% even when every sandbox test passed.

These scripts diagnose [issue #39](https://github.com/AI-Security-Bootcamp/aisb/issues/39)
without modifying the curriculum or installed libraries. See
[RESULTS.md](RESULTS.md) for the real-model run and its limitations.

## Deterministic reproduction

Activate an environment with `requirements-control-arena.txt` installed, then
run from the repository root:

```bash
python reproductions/issue_39/reproduce.py
python reproductions/issue_39/reproduce.py --bounded \
  --output reproductions/issue_39/artifacts/deterministic-bounded
```

No API credentials or Docker are needed. The script uses the actual
`evaluate_monitor_on_trajectories` helper and ControlArena monitor machinery,
with 100 synthetic saved trajectories and a simulated model response:

- The baseline completes 99 responses and holds the last behind an async gate.
  Assertions verify 99% progress, a pending monitor runner, and zero AUC calls.
  Releasing the response lets the real helper save scores and produce a plot.
- `--bounded` adds `GenerateConfig(attempt_timeout=1, max_retries=0)`. Assertions
  verify that the timeout ends the wait but raises an exception containing
  `AttemptTimeoutError`. AUC never runs, and no scored logs are saved.
- Both variants wait longer than the original saved evaluations' one-second
  sample limit. That limit is not applied to this later monitoring operation.

The synthetic AUC is only a plumbing check, not a measurement of monitor quality.
Artifacts include `results.json`, captured progress, saved fixture logs, and a
baseline ROC plot. Use a new output directory when comparing different versions.
The script imports diagnostic helpers from `../issue_38/`.

## Real APPS reproduction

This requires the Docker setup used by the course and OpenRouter credentials in
an existing environment file. Replace the example environment-file path below.

```bash
python reproductions/issue_38/reproduce_apps.py replay \
  --limit 50 --env-file /path/to/credentials.env \
  --output reproductions/issue_39/artifacts/apps-replay-100

timeout 600 python -u reproductions/issue_39/run_live.py \
  --env-file /path/to/credentials.env \
  --input reproductions/issue_39/artifacts/apps-replay-100 \
  --output reproductions/issue_39/artifacts/live-post-hoc
```

The first command replays the dataset's supplied honest and backdoored solutions
through the real APPS sandbox/scorer (50 samples per mode). The second runs the
unchanged course monitor prompt, model, cache, and concurrency configuration over
those trajectories. It records each monitor's mode, sample ID, start, completion,
score, and elapsed time in `trace.jsonl`, plus phase boundaries around monitoring,
AUC, and plotting. It uses Matplotlib's `Agg` backend to save plots without a GUI.

The external ten-minute process guard is a diagnostic budget, not a proposed
fix. It does not provide per-request error handling or preserve scored eval logs
when the course helper remains blocked. Trace records preserve completed scores
for this investigation only. A monitor's elapsed time includes any queueing and
retries; the trace does not identify the underlying network failure.

## Mitigation requirements

Request deadlines and bounded retries prevent a single generation from waiting
indefinitely, but the current helper propagates a failed call and aborts the
batch. A complete course fix should also:

1. Catch failures per sample, record the sample ID and failure explicitly, and
   checkpoint completed scores as they arrive.
2. Support retrying only failed or unfinished samples.
3. Report successful, failed, and pending counts. If displaying an interim ROC,
   label it incomplete and state coverage for both honest and attack samples;
   missing scores must not become zero scores or silently disappear from the
   reported experiment.
4. Preserve monitor request/response/error events for inspection, addressing the
   separate logging problems documented for issue #38.

The timeout-only variant above deliberately retains the existing error behavior
so its limitations remain reproducible. No production fix is included here.

## Compare sample sizes and retry selected inputs

```bash
python reproductions/issue_39/check_sample_sizes.py \
  --input reproductions/issue_39/artifacts/apps-replay-100 \
  --trace reproductions/issue_39/artifacts/live-post-hoc/trace.jsonl \
  --output reproductions/issue_39/artifacts/size-check
```

This reconstructs the original monitor prompts and compares their lengths with
the original completion trace. Add `--env-file /path/to/credentials.env` to make
two uncached model calls for each previously stalled prompt and each of the four
largest completed prompts. These follow-up calls have concurrency four and
explicit deadlines, so they test repeatability of the inputs rather than
replicating the original batch's full concurrency. Use a fresh output directory
for each live run. The script saves raw outputs and reported token usage to
separate response completion from valid score formatting.
