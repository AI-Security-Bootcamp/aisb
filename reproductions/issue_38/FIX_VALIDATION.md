# Issue 38 fix validation — 2026-09-16

Branch: `fix/38-monitor-request-logging`, based on `3bc387e`.

## Before and after

The unchanged Inspect/ControlArena path still reproduces both logging defects:
`reproduce.py` loses post-hoc monitor events, and `reproduce_timeout.py` leaves a
cancelled live request pending. The pinned Inspect viewer hides that request.

The course now evaluates saved trajectories inside native Inspect samples, so
each monitor conversation has a recorded transcript. A scoped compatibility
wrapper finalizes interrupted model events with an explicit error and propagates
the original interruption. It does not create a response or a score. Reference
and scaffold prompts now use the parser's `<reasoning>` tag.

| Fresh dev-box check | Result |
| --- | --- |
| Real APPS live run, 10 honest + 10 attack | 20 monitor requests retained; 19 responses; one real 120-second sample timeout |
| Timed-out honest sample 31 | Request retained with `pending=false`, `Model call interrupted: CancelledError`, and no response |
| Real post-hoc monitoring, 2 honest + 2 attack | Four requests, four responses, four scores persisted; zero errors |
| Local deterministic regression suite | 4 passed |
| Dev-box deterministic regression suite | 4 passed |
| Instruction build | Passed; generated Markdown and tests rebuilt from the solution |

Browser verification of the real sample 31 under the default Transcript event
filter showed its original monitor prompt and the cancellation error. The same
check passed for the deterministic timeout. The original unwrapped reproduction
still hides the pending request, isolating the behavior changed by this patch.

The regression suite additionally checks cache hits, source-log immutability,
score preservation, and cancellation isolation between concurrent monitors.

## Repeat the checks

```bash
python -m pytest 2.2-monitoring/tests/test_monitor_logging.py -q
./build-instructions.sh 2.2-monitoring/section2_solution.py
python reproductions/issue_38/reproduce_timeout.py
python reproductions/issue_38/reproduce.py
```

For real APPS checks, use the `live` and `post-hoc` commands in this folder's
README with an existing credential environment file. Fresh artifacts are under
`artifacts/fixed-live-20/` and `artifacts/fixed-post-hoc-4/` locally and under
`/root/aisb-fix38/reproductions/issue_38/artifacts/` on the dev box. Credentials
and generated eval logs are not committed.

Dependencies remain ControlArena 17.1.1, Inspect 0.3.244, and OpenAI client 2.54.0.
Tests passed on Python 3.11.2 locally and Python 3.12.14 remotely. The compatibility
wrapper uses Inspect's private `_event_updated` method to synchronize the saved
event; the regression checks should be rerun before changing the Inspect pin.

This branch addresses request visibility. Time limits, valid-score coverage,
and resumable batch recovery are the separate stacked fix for issue #39.
