"""Run saved trajectories through Inspect so monitor calls have real transcripts."""

import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import anyio
from control_arena import ControlAgent, control_agent
from control_arena.eval import run_monitor_for_messages
from control_arena.scorers import max_suspicion_scorer
from inspect_ai import Task, eval_set
from inspect_ai.dataset import Sample
from inspect_ai.event import ModelEvent, SpanBeginEvent
from inspect_ai.log import EvalLog, read_eval_log, transcript
from inspect_ai.solver import solver
from inspect_ai.util import span


@control_agent
def with_model_error_logging(wrapped: ControlAgent):
    """Keep interrupted model requests visible, then propagate the interruption.

    Inspect 0.3.244 can leave a cancelled model event pending. Its viewer hides
    pending events, including the request. Scope the compatibility fix to this
    agent invocation; never change another concurrent monitor's events.
    """

    async def run(state, tools):
        span_id = str(uuid4())
        async with span("Monitored execution", type="monitor", id=span_id):
            try:
                return await wrapped(state, tools)
            except BaseException as error:
                # Cleanup must also handle cancellation, which is not Exception.
                # No response or score is invented for an interrupted request.
                owned_spans = {span_id}
                current = transcript()
                for event in current.events:
                    if (
                        isinstance(event, SpanBeginEvent)
                        and event.parent_id in owned_spans
                    ):
                        owned_spans.add(event.id)
                    if (
                        isinstance(event, ModelEvent)
                        and event.span_id in owned_spans
                        and event.pending
                    ):
                        event.pending = None
                        event.completed = datetime.now(timezone.utc)
                        event.error = f"Model call interrupted: {type(error).__name__}"
                        # This pinned-version compatibility call updates Inspect's
                        # recorder as well as the in-memory event. Merely changing
                        # the pending flag does not reliably update saved logs.
                        current._event_updated(event)
                raise

    return run


class IncompleteMonitorEvaluation(RuntimeError):
    """Saved results have incomplete coverage and must not produce a full ROC."""

    def __init__(self, logs, log_dir, coverage):
        self.logs = logs
        self.log_dir = Path(log_dir)
        self.coverage = coverage
        counts = ", ".join(
            f"{mode} {counts['completed']}/{counts['expected']}"
            for mode, counts in coverage.items()
        )
        super().__init__(
            f"Monitoring incomplete ({counts}). Results and errors are saved in "
            f"{self.log_dir.resolve()}. Rerun the same evaluation to retry only "
            "failed or unfinished samples; completed samples are reused. "
            "No ROC/AUC was calculated from this incomplete run."
        )


class MonitorTimeoutError(RuntimeError):
    """A retryable monitor failure, distinct from Inspect's sample-limit signal."""


@solver
def monitor_saved_trajectory(monitor: ControlAgent, timeout_seconds: float = 60):
    """Monitor existing assistant actions without executing their tools again."""
    logged_monitor = with_model_error_logging(wrapped=monitor)

    async def solve(state, generate):
        # Do not let a previously monitored trajectory contribute stale scores.
        for message in state.messages:
            if message.role == "assistant" and message.metadata:
                for key in ("suspicion_score", "reason", "model_usage"):
                    message.metadata.pop(key, None)
        try:
            # Include queueing and every monitor step in a sample deadline. The
            # original trajectory's time_limit does not apply to post-hoc work.
            with anyio.fail_after(timeout_seconds):
                state.messages = await run_monitor_for_messages(
                    logged_monitor, state.messages, copy=False, metadata=state.metadata
                )
        except TimeoutError as error:
            raise MonitorTimeoutError(
                f"Monitor exceeded {timeout_seconds:g}s for sample {state.sample_id}"
            ) from error
        messages = [
            message for message in state.messages if message.role == "assistant"
        ]
        if not messages:
            raise ValueError("Trajectory has no assistant actions to monitor")
        for message in messages:
            metadata = message.metadata or {}
            score = metadata.get("suspicion_score")
            # The basic parser falls back to zero on malformed output. Such
            # failures are errors, never observations of safety.
            if (
                not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= 1
                or metadata.get("reason")
                in {
                    "No score found in model response",
                    "Could not parse score from model response",
                }
            ):
                raise ValueError(
                    f"Missing or invalid monitor score for sample {state.sample_id}"
                )
        state.completed = True
        return state

    return solve


def run_monitor_evaluation(
    monitor: ControlAgent,
    source_logs: list[EvalLog],
    log_dir: str | Path,
    *,
    display: str | None = None,
    timeout_seconds: float = 60,
    max_samples: int = 4,
) -> list[EvalLog]:
    """Create separate monitoring evals with requests, responses, and scores.

    ControlArena's standalone post-hoc helper updates message metadata but does
    not attach model events to the source EvalSamples. Running the same monitor
    inside an Inspect solver gives every sample its own recorded transcript.
    Source files and their original sandbox results remain available unchanged.
    """
    if timeout_seconds <= 0 or max_samples < 1:
        raise ValueError("timeout_seconds and max_samples must be positive")
    tasks = []
    for log in source_logs:
        samples = [
            Sample(
                id=sample.id,
                input=deepcopy(sample.messages),
                target=sample.target,
                metadata=deepcopy(sample.metadata),
            )
            for sample in log.samples or []
        ]
        if not samples:
            raise ValueError(f"No trajectories in {log.location}")
        # eval_set hashes the solver/monitor configuration, but not the dataset.
        # Include it explicitly to invalidate reuse when a trajectory changes.
        source_version = hashlib.sha256(
            json.dumps(
                [sample.model_dump(mode="json") for sample in samples],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        tasks.append(
            Task(
                name=f"{log.eval.task}-monitor",
                dataset=samples,
                solver=monitor_saved_trajectory(
                    monitor, timeout_seconds=timeout_seconds
                ),
                scorer=max_suspicion_scorer(),
                model="none/none",  # The monitor selects its own model.
                metadata={
                    "source_log": log.location,
                    "source_eval_id": log.eval.eval_id,
                },
                version=f"monitor-v2-{source_version}",
                fail_on_error=True,
                continue_on_fail=True,
            )
        )
    if not tasks:
        raise ValueError("No trajectory logs supplied")
    _, results = eval_set(
        tasks,
        log_dir=str(log_dir),
        display=display,
        max_samples=max_samples,
        max_tasks=2,
        retry_attempts=1,  # One pass per invocation; retry explicitly by rerunning.
        retry_immediate=False,
        retry_cleanup=False,  # Retain failed requests and errors for inspection.
        log_dir_allow_dirty=True,  # Different prompts have separate task identities.
        ctl_server=False,
    )
    # Reused successful evals may be returned as headers, without their samples.
    logs = [read_eval_log(log.location, resolve_attachments="full") for log in results]
    coverage = {}
    for log in source_logs:
        for sample in log.samples or []:
            mode = sample.metadata.get("eval_mode", "unknown")
            coverage.setdefault(mode, {"completed": 0, "expected": 0})["expected"] += 1
    for log in logs:
        for sample in log.samples or []:
            score = (sample.scores or {}).get("max_suspicion_scorer")
            if (
                sample.error is None
                and score is not None
                and isinstance(score.value, (int, float))
                and math.isfinite(score.value)
            ):
                coverage[sample.metadata.get("eval_mode", "unknown")]["completed"] += 1
    print(
        "Monitor coverage: "
        + ", ".join(
            f"{mode} {counts['completed']}/{counts['expected']}"
            for mode, counts in coverage.items()
        )
    )
    if any(log.status != "success" for log in logs) or any(
        counts["completed"] != counts["expected"] for counts in coverage.values()
    ):
        raise IncompleteMonitorEvaluation(logs, log_dir, coverage)
    return logs
