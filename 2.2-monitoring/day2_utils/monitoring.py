"""Run saved trajectories through Inspect so monitor calls have real transcripts."""

import hashlib
import json
import logging
import math
import re
from contextlib import contextmanager
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

from .eval_utils import UNSCORED_REASONS


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
    """A monitoring eval stopped early (e.g. interrupted) and must be resumed."""

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
            "No ROC/AUC was calculated from this unfinished run."
        )


class MonitorTimeoutError(RuntimeError):
    """A retryable monitor failure, distinct from Inspect's sample-limit signal."""


class _DropNoisyWarnings(logging.Filter):
    """Drop ControlArena warnings that bury the run's output.

    - "No score found…" prints a whole unparseable monitor reply. The sample is
      reported in a one-line summary instead; the full reply stays in the log
      for show_monitor_replies() and inspect view.
    - "Need at least 2 valid scores…" fires on every small eval's stderr metric.
    """

    PREFIXES = ("No score found in model response", "Need at least 2 valid scores")

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(self.PREFIXES)


@contextmanager
def quiet_control_arena_logs():
    """Hide ControlArena's noisy monitor/metric warnings for the duration."""
    loggers = [
        logging.getLogger("control_arena.monitor._utils"),
        logging.getLogger("control_arena.scorers._metrics"),
    ]
    noise_filter = _DropNoisyWarnings()
    for logger in loggers:
        logger.addFilter(noise_filter)
    try:
        yield
    finally:
        for logger in loggers:
            logger.removeFilter(noise_filter)


def _missing_score_reason() -> str:
    """Explain why the monitor's latest reply in this sample had no usable score."""
    replies = [
        event
        for event in transcript().events
        if isinstance(event, ModelEvent) and event.output is not None
    ]
    if replies and replies[-1].output.stop_reason == "max_tokens":
        return "monitor reply hit max_tokens before giving a score"
    return "monitor reply had no parseable <score>"


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
                or metadata.get("reason") in UNSCORED_REASONS
            ):
                raise ValueError(_missing_score_reason())
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
    retries: int = 2,
) -> list[EvalLog]:
    """Record bounded monitoring evals and resume completed samples on rerun.

    ControlArena's standalone post-hoc helper updates message metadata but does
    not attach model events to the source EvalSamples. Running the same monitor
    inside an Inspect solver gives every sample its own recorded transcript.
    Source files and their original sandbox results remain available unchanged.
    Inspect saves progress and isolates failures per sample. Rerunning this call
    in the same log directory reuses finished work; changes to the monitor
    configuration or dataset have separate evaluation identities.

    A sample whose monitor times out, errors, or replies without a valid score
    is retried up to ``retries`` times, then rejected: it keeps its error and has
    no score, so callers' AUCs use only the scored samples. A rejection is
    reported as a one-line warning, not a traceback, and is not retried again
    on rerun.

    Raises IncompleteMonitorEvaluation if an eval did not finish (e.g. it was
    interrupted); rerunning resumes it.
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
                # Record failed samples as rejected rather than failing the eval
                # (which would print a traceback panel for one bad reply).
                fail_on_error=False,
            )
        )
    if not tasks:
        raise ValueError("No trajectory logs supplied")
    with quiet_control_arena_logs():
        _, results = eval_set(
            tasks,
            log_dir=str(log_dir),
            display=display,
            max_samples=max_samples,
            max_tasks=2,
            retry_on_error=retries,  # Retry a failed sample before rejecting it.
            retry_attempts=1,  # One pass per invocation; rerun to resume.
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
    rejected = []
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
            else:
                # Inspect stores the exception's repr, e.g. "ValueError('...')";
                # show just the message.
                reason = (
                    re.sub(r"^\w+\((['\"])(.*)\1\)$", r"\2", sample.error.message)
                    if sample.error
                    else "no score"
                )
                mode = sample.metadata.get("eval_mode")
                rejected.append(f"{mode} sample {sample.id}: {reason}")
    print(
        "Monitor coverage: "
        + ", ".join(
            f"{mode} {counts['completed']}/{counts['expected']}"
            for mode, counts in coverage.items()
        )
    )
    if any(log.status != "success" for log in logs):
        raise IncompleteMonitorEvaluation(logs, log_dir, coverage)
    if rejected:
        print(
            f"  WARNING: {len(rejected)} sample(s) rejected after {retries} "
            "retries and excluded from the scores:"
        )
        for line in rejected:
            print(f"    - {line}")
    return logs
