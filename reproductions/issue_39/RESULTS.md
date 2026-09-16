# Issue 39 reproduction results — 2026-09-16

**Reproduced a late monitoring stall with real model calls, and the exact 99%
symptom with a controlled response.** The stalled phase precedes ROC/AUC
calculation. Adding only a request timeout changes the controlled stall into a
batch error; it does not let the course save the completed scores or draw a ROC.

## Real APPS run

Used the actual section 2.2 helpers and reference monitor prompt, with 50 honest
and 50 attack trajectories produced by replaying the dataset's supplied solutions
through the real Docker sandbox. The monitor was the course's
`openrouter/meta-llama/llama-3.1-8b-instruct`, with the original cache and
concurrency settings. No course timeout/retry configuration was added.

| Stage | Observed result |
| --- | --- |
| APPS replay | 100/100 main tasks passed; 50/50 attack side tasks passed; no sample errors or sample-limit events |
| Post-hoc monitoring | 100 started; 96 completed by 76.874 seconds; four remained unfinished |
| Remaining wait | No further completions before the external ten-minute process guard stopped the run, exit code 124 |
| AUC and plotting | Neither phase started; no scored eval logs or plots were saved by the course helper |

The unfinished monitor evaluations were attack sample `97`, and honest samples
`116`, `197`, and `36`. All four had already passed their sandbox tests. The
monitoring run began at `09:28:41.724816 UTC`; its last completion was at
`09:29:58.595605 UTC`. Progress stayed at 96% for approximately another eight and
a half minutes until the diagnostic guard stopped the process.

This real run reproduced a stall near completion, **not exactly 99%**. The trace
locates the unfinished work in monitor calls; it does not distinguish provider
latency, transport waiting, or retries within those calls. It does not prove that
the wait would never end. No broken sandbox example was found in these 100
evaluations, and the evidence does not justify swapping a dataset sample.

## Exact 99% reproduction and timeout check

`reproduce.py` loads the actual course evaluation helper and creates 100 synthetic
saved trajectories. A simulated model finishes 99 responses and holds the last
behind an async gate. All monitoring, progress, scoring, and plotting machinery
otherwise uses the installed course dependencies.

| Variant | Result while the final response is held | Result afterward |
| --- | --- | --- |
| Original configuration | 99/100 progress; monitor runner pending; zero AUC calls | Releasing the response completes scoring and produces a ROC plot |
| One-second attempt timeout, zero retries | 99/100 progress; runner ended with an exception containing `AttemptTimeoutError`; zero AUC calls | Batch fails; zero scored eval logs saved |

Both variants passed locally on Python 3.11.2 and on the dev box on Python
3.12.14. In the baseline, plotting after release took 0.23 seconds locally and
0.298 seconds remotely. The fixtures' original sample time limit was one second;
monitoring remained pending after a two-second hold, demonstrating that this
saved evaluation limit does not govern subsequent post-hoc monitoring.

## Why it happens

In [section2_solution.py](../../2.2-monitoring/section2_solution.py),
`evaluate_monitor_on_trajectories` awaits `run_monitor_for_eval_logs` before
rescoring, calculating AUC, or writing scored logs. Plotting happens after the
helper returns.

In ControlArena 17.1.1, `eval/_run_monitors.py` advances progress only after a
monitor returns and waits for all calls through nested `gather` operations. This
path does not apply the original evaluation's sample time limit. The course's
`create_monitor` supplies no explicit request deadline or retry limit. Inspect
0.3.244's default generation configuration permits unlimited retries, although
this trace does not establish whether retries caused the four unfinished calls.

ControlArena's `gather` uses an AnyIO task group. An uncaught monitor error aborts
that group and propagates instead of returning the successful subset. That is
why adding only `attempt_timeout` and `max_retries` stops waiting but still
prevents the existing helper from producing results.

The mitigation needs deadlines plus per-sample failure handling, incremental
score persistence, and a way to retry unfinished samples. Any interim ROC must
show that it is incomplete and report coverage by class; timeouts must not be
silently assigned zero suspicion. Monitor event retention remains a separate
requirement from [issue #38](../issue_38/DEVBOX_RESULTS.md).

## Environment, artifacts, and limits

- Dev-box checkout: `3bc387e91ddadecc7e8177cc7546168186b5406a`.
- Python 3.12.14; ControlArena 17.1.1; Inspect 0.3.244; OpenAI client 2.54.0.
- Docker 26.1.5, Compose 2.26.1, and the course's `python:3.4` sandbox image.
- `artifacts/devbox/apps-replay-100/audit.json`: source hashes, environment, and
  every sample's sandbox result, error, and limit events.
- `artifacts/devbox/live-post-hoc/trace.jsonl`: per-monitor starts, completions,
  scores, and phase boundaries. The diagnostic trace retains 96 completed
  scores even though the course helper did not save scored eval logs.
- `artifacts/devbox/summary.json`: aggregate results and unfinished sample IDs.
- `artifacts/{deterministic,deterministic-bounded}/results.json` and their dev-box
  equivalents: assertion-backed checks for the exact 99% case and deadline.

The original on-site logs and environment were unavailable. This establishes a
reproducible failure mechanism with the current pinned environment, rather than
proving the precise trigger of every reported classroom incident. No curriculum,
dependency pins, or installed library code were changed. See [README.md](README.md)
for commands to repeat the tests.

## Follow-up: are particular samples too large?

The size comparison and fresh retries argue against a deterministic oversized
input failure. Reconstructed all 100 monitor prompts using ControlArena's actual
monitor-view conversion and prompt formatter. Prompt sizes ranged from 2,112 to
5,869 characters, with a median of 3,775.5. The stalled set included both the
largest prompt and two of the smallest 14 prompts.

| Previously stalled sample | Prompt characters | Size rank, smallest = 1 | Retry 1 | Retry 2 |
| --- | ---: | ---: | ---: | ---: |
| Attack 97 | 5,869 | 100/100 | 3.156 s | 5.628 s |
| Honest 116 | 2,842 | 8/100 | 4.489 s | 9.756 s |
| Honest 197 | 5,337 | 92/100 | 20.262 s | 0.996 s |
| Honest 36 | 2,965 | 14/100 | 14.498 s | 13.565 s |

All four returned in both uncached retry rounds. Their reported input usage was
747–1,826 tokens. The four largest previously completed prompts were used as
controls (honest 147, honest 72, attack 197, attack 19); these also returned in
both rounds, in 3.748–22.541 seconds. All 16 responses ended with stop reason
`stop`. One response (honest 197, round 2) lacked a score tag, so response
completion must not be interpreted as 16 valid monitor scores.

The retries sent the reconstructed course prompts directly through Inspect's
model API, with the same model and no output-token cap added. They disabled
caching, limited concurrency to four, and used a 60-second attempt deadline,
zero retries, and a 75-second outer call deadline. No deadline fired. These
changes and the later execution time mean the experiment does **not** isolate
provider variability from concurrency, or rule out probabilistic effects of
prompt content. It does show that none of the four inputs invariably stalls.

For issue #38, the timed-out honest sample 37's actual saved monitor request was
3,175 characters: fourth-smallest of 20, whose median was 4,331 and maximum was
5,975 characters. This is another counterexample to a simple large-input
explanation. That particular live request was not retried in this follow-up.

The original issue #39 run also contained completed responses of up to 5,339
output tokens. Input length alone therefore does not characterize the amount
of generation work; output length for the four unfinished calls is unknown.

Evidence: `artifacts/devbox/size-check/{size-summary.json,all-sizes.json,
retry-results.json}`, plus the saved raw model responses. The script is
`check_sample_sizes.py`. No dataset examples were removed or changed.
