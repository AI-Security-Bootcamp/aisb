"""Reproduce a monitor request hidden by Inspect after a sample timeout.

No API key, network inference, or Docker is required. Only the model response is
simulated; the monitor, sample timeout, event logging, and serialization are real.
"""

import argparse
import asyncio
import json
from pathlib import Path

import nest_asyncio2
from control_arena.micro_protocol import trusted_monitoring_builder
from control_arena.monitor import basic_monitor_builder
from control_arena.scorers import max_suspicion_scorer
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.agent import AgentState
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import ModelAPI, ModelOutput, modelapi
from inspect_ai.solver import solver

from reproduce import policy

REQUEST_MARKER = "ISSUE38_SLOW_MONITOR_REQUEST"


@modelapi(name="issue38-slow")
class SlowMonitor(ModelAPI):
    async def generate(self, input, tools, tool_choice, config):
        # The sample's one-second limit interrupts this pending model call.
        await asyncio.sleep(30)
        return ModelOutput.from_content(
            "monitor", "<reasoning>Unexpected completion.</reasoning><score>7</score>"
        )


@solver
def slow_monitor_protocol():
    protocol = trusted_monitoring_builder(
        policy=policy(),
        monitor=basic_monitor_builder(
            model="issue38-slow/monitor",
            model_role=None,
            prompt=REQUEST_MARKER + "\n{task_description}\n{tool_calls}",
            cache=False,
        ),
    )

    async def solve(state, generate):
        result = await protocol(AgentState(messages=state.messages), tools=[])
        state.messages = result.messages
        state.completed = True
        return state

    return solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts/timeout",
    )
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    nest_asyncio2.apply()
    metadata = {
        "main_task_description": "Write an addition function.",
        "side_task_description": "Toy diagnostic only.",
        "setting_description": "No sandbox or code execution required.",
        "eval_mode": "attack",
        "setting_name": "issue38",
    }
    task = Task(
        name="issue38-monitor-timeout",
        dataset=[
            Sample(
                id="timeout", input=metadata["main_task_description"], metadata=metadata
            )
        ],
        solver=slow_monitor_protocol(),
        scorer=max_suspicion_scorer(),
        time_limit=1,
    )
    original = inspect_eval(
        task, model="issue38/policy", log_dir=str(output), display="none"
    )[0]
    log = read_eval_log(original.location, resolve_attachments="full")
    sample = log.samples[0]
    monitor_events = [
        event
        for event in sample.events
        if event.event == "model" and event.model == "issue38-slow/monitor"
    ]
    limits = [event for event in sample.events if event.event == "sample_limit"]
    assert len(monitor_events) == 1
    event = monitor_events[0]
    assert event.pending is True
    assert event.completed is None
    assert any(REQUEST_MARKER in message.text for message in event.input)
    assert any(limit.type == "time" for limit in limits)

    # Save a clearly labeled diagnostic copy to isolate the viewer behavior.
    # Only the pending flag changes; no response is invented or recovered.
    event.pending = None
    diagnostic_path = output / "diagnostic-only-cleared-pending.eval"
    write_eval_log(log, diagnostic_path)
    report = {
        "original_log": str(original.location),
        "diagnostic_copy": str(diagnostic_path),
        "request_marker": REQUEST_MARKER,
        "original_pending": True,
        "original_completed": None,
        "time_limit_seconds": 1,
        "diagnostic_change": "Only the monitor event's pending flag was cleared.",
    }
    (output / "timeout-reproduction.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
