# Issue 38: missing monitor requests in Inspect

**Latest result (2026-09-15): reproduced a missing live monitor request after a
sample timeout.** The raw event remains marked `pending`, and Inspect's viewer
filters it out. See the [full APPS results](DEVBOX_RESULTS.md) and the small
[`reproduce_timeout.py`](reproduce_timeout.py) reproduction, which requires no
API key or Docker.

The saved-trajectory monitoring path used in Exercise 2.2.1 separately loses the
monitor's model-call events. Completed live calls and cache hits retain their
requests and responses. The original on-site logs are unavailable, so the cause
of that particular [incident](https://github.com/AI-Security-Bootcamp/aisb/issues/38)
remains unconfirmed.

## Initial deterministic checks (2026-09-14)

Tested with Python 3.11.2 and the course pins: `control-arena==17.1.1`,
`inspect-ai==0.3.244`, and `openai==2.54.0`. Dependencies were installed in an
isolated environment from `requirements-control-arena.txt`.

| Path | Suspicion score | Monitor model events in saved log | Full request and response |
| --- | --- | --- | --- |
| Original trajectory, no monitor | N | 0 | Not applicable |
| Monitor saved trajectory, rescore, save | 0.7 | 0 | Missing |
| Live `trusted_monitoring_builder` | 0.7 | 1 | Present |
| Live monitor, cache hit | 0.7 | 1 | Present |

The saved-trajectory log retains the **parsed score and reasoning** in the
assistant message's metadata. It loses the separate monitor model event, including
the prompt and full model response. Assertions check results after reading the
`.eval` files back from disk, with deduplicated attachment content resolved.

Browser verification with the pinned Inspect viewer confirmed:

- The saved-trajectory log's **Transcript** contains only the policy model call,
  while its suspicion score and parsed reasoning remain visible as metadata.
- The live log's **Transcript** displays both model calls, including the monitor
  prompt and full response, with the default event filter.
- The live log's **Messages** tab contains the agent conversation and monitor
  metadata, without the separate monitor conversation. Use **Transcript** to
  inspect that model call.

## Reproduce

Run from the repository root, using the course ControlArena environment:

```bash
python -m venv .venv-control-arena
.venv-control-arena/bin/python -m pip install -r requirements-control-arena.txt
.venv-control-arena/bin/python reproductions/issue_38/reproduce.py
.venv-control-arena/bin/inspect view --log-dir reproductions/issue_38/artifacts/logs
```

The script prints its dependency versions and results, writes
`artifacts/results.json`, and saves four diagnostic `.eval` files. An alternative
output directory can be passed with `--output`. Generated artifacts are ignored
by Git. Each run uses a fresh monitor prompt identifier so the cache-hit check
does not depend on previous executions.

Open `issue38-post-hoc.eval` and either `issue38-live` log, select the single sample,
then compare **Transcript**. The post-hoc file retains the original task name
`issue38-original`; distinguish it by its filename. Inspect may hide the original
unscored file as a retried run because it shares the post-hoc file's task ID.

All assertions passed. They characterize the current bug: a future fix that saves
post-hoc monitor events should fail the assertion expecting zero such events.

## Cause of the reproduced case

The course's
[`evaluate_monitor_on_trajectories`](../../2.2-monitoring/section2_solution.py)
uses this sequence:

```python
run_monitor_for_eval_logs(monitor, eval_logs)
inspect_score(log, [max_suspicion_scorer()], action="overwrite")
write_eval_log(log, path)
```

In ControlArena 17.1.1,
`control_arena/eval/_run_monitors.py:152` (`run_monitor_for_sample`) copies the
sample and updates `sample.messages` with monitor metadata, then returns without
adding monitor events to `sample.events`. In Inspect 0.3.244,
`inspect_ai/_eval/score.py:430` initializes scoring's transcript from the sample's
existing events. Rescoring therefore preserves the omission; it cannot reconstruct
the missing monitor call.

The live protocol invokes the monitor inside the active Inspect evaluation, so
normal model-call recording captures it. Both an actual deterministic model call
and a cache hit survive serialization.

The course helper's comments also say the monitor appends a new message and that
the scorer reads monitor messages. This version of ControlArena instead annotates
existing assistant messages with `suspicion_score` and `reason` metadata.

## Implications and limits

For the reproduced saved-trajectory case, investigate recording each monitor call
inside an Inspect sample transcript and persisting those events with that sample.
A viewer filter cannot restore events that were never saved. These results do not
establish that an Inspect version bump is necessary or sufficient.

For the live exercise, clarify **Transcript** versus **Messages** in the
instructions. If an affected on-site `.eval` file is available, check whether its
monitor model event exists before attributing the incident to logging or display.

This diagnostic uses the real ControlArena monitor and protocol, Inspect scoring,
model logging, caching, and serialization. A custom Inspect `ModelAPI` supplies
deterministic policy and monitor responses, and the submitted tool call is not
executed. No API key, model service, APPS dataset, or Docker sandbox is required.
The initial diagnostic did not run the full APPS/OpenRouter exercise. The later
dev box follow-up below tests the actual provider, sandbox, and concurrent live
exercise, including its timeout behavior.

The workspace was at `06bb2c5`; the relevant 2.2/2.3 content and
`requirements-control-arena.txt` matched remote main `3bc387e` when checked.
Only this diagnostic and its report were added; curriculum and dependency pins
were not changed.

## Full APPS diagnostic

The [dev box follow-up](DEVBOX_RESULTS.md) records the real Docker/APPS replay,
live monitoring, cache checks, and the reproduced timeout failure.

[`reproduce_apps.py`](reproduce_apps.py) exercises the actual course helper with
the APPS dataset and Docker. It loads the three relevant function definitions
directly from `section2_solution.py` so unrelated exercises do not run, and uses
the unchanged reference attack and monitor prompts. Live mode retains the course
defaults of ten samples per mode, eight sandboxes, two tasks, and a 120-second
per-sample time limit.

With the ControlArena environment activated, run:

```bash
# Check the real sandbox with two dataset solutions per mode.
python reproductions/issue_38/reproduce_apps.py replay --limit 2 \
  --output reproductions/issue_38/artifacts/apps-replay

# Requires OPENROUTER_API_KEY; these commands make paid model calls.
python reproductions/issue_38/reproduce_apps.py post-hoc \
  --input reproductions/issue_38/artifacts/apps-replay \
  --output reproductions/issue_38/artifacts/apps-post-hoc
python reproductions/issue_38/reproduce_apps.py live \
  --output reproductions/issue_38/artifacts/apps-live
```

Pass `--env-file /path/to/credentials.env` if credentials are outside the checkout.
Each output directory contains the original Inspect `.eval` files and `audit.json`.
The audit records source hashes, the commit and package versions, sample errors,
scores, model-event counts, cache status, whether monitor inputs and outputs are
nonempty, and the parsed monitor metadata. It can be regenerated without model
calls using `audit --output /path/to/existing/logs`.

A separate parser check on the dev box with ControlArena 17.1.1 confirmed a
reference-prompt mismatch: the course asks for `<reason>` tags, whereas the basic
monitor parser defaults to `<reasoning>`. An otherwise valid response with
`<reason>Example monitor explanation.</reason><score>7</score>` produces score 0.7
and `reason="No reasoning provided."`. Changing only the tag to `<reasoning>`
preserves the explanation. This affects parsed metadata; it does not establish
that the separate model event is missing.
# Fix branch

See [FIX_VALIDATION.md](FIX_VALIDATION.md) for the implemented fix and fresh
before/after validation. The investigation below records the original defects;
the standalone baseline reproductions intentionally still exercise the unchanged
upstream functions. `reproduce_apps.py` uses the course code on the current branch.
