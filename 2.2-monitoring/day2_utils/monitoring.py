"""Run saved trajectories through Inspect so monitor calls have real transcripts."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from control_arena import ControlAgent, control_agent
from control_arena.eval import run_monitor_for_messages
from control_arena.scorers import max_suspicion_scorer
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.event import ModelEvent, SpanBeginEvent
from inspect_ai.log import EvalLog, transcript
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


@solver
def monitor_saved_trajectory(monitor: ControlAgent):
    """Monitor existing assistant actions without executing their tools again."""
    logged_monitor = with_model_error_logging(wrapped=monitor)

    async def solve(state, generate):
        state.messages = await run_monitor_for_messages(
            logged_monitor, state.messages, copy=False, metadata=state.metadata
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
) -> list[EvalLog]:
    """Create separate monitoring evals with requests, responses, and scores.

    ControlArena's standalone post-hoc helper updates message metadata but does
    not attach model events to the source EvalSamples. Running the same monitor
    inside an Inspect solver gives every sample its own recorded transcript.
    Source files and their original sandbox results remain available unchanged.
    """
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
        tasks.append(
            Task(
                name=f"{log.eval.task}-monitor",
                dataset=samples,
                solver=monitor_saved_trajectory(monitor),
                scorer=max_suspicion_scorer(),
                model="none/none",  # The monitor selects its own model.
                metadata={
                    "source_log": log.location,
                    "source_eval_id": log.eval.eval_id,
                },
            )
        )
    if not tasks:
        raise ValueError("No trajectory logs supplied")
    return inspect_eval(tasks, log_dir=str(log_dir), display=display)
