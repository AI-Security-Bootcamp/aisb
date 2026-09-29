"""Issue #38: requests survive saved-trajectory monitoring and cancellation."""

import asyncio
from copy import deepcopy

import pytest
from control_arena.micro_protocol import trusted_monitoring_builder
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.agent import AgentState
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.solver import solver

from day2_utils.monitoring import run_monitor_evaluation, with_model_error_logging
from .fixtures import CALLS, SLOW, make_monitor, make_source_logs, submit_policy


def monitor_events(sample):
    return [
        event
        for event in sample.events
        if event.event == "model" and event.model == "aisb-monitor-test/monitor"
    ]


def test_saved_monitor_requests_responses_and_cache(tmp_path):
    sources = make_source_logs(tmp_path / "source")
    original = [deepcopy(log) for log in sources]
    monitor = make_monitor(cache=True)
    first = run_monitor_evaluation(monitor, sources, tmp_path / "first", display="none")
    second = run_monitor_evaluation(
        monitor, sources, tmp_path / "second", display="none"
    )
    assert sources == original, "Monitoring must not mutate the source logs"
    for run, expected_cache in [(first, "write"), (second, "read")]:
        for result in run:
            saved = read_eval_log(result.location, resolve_attachments="full")
            assert saved.eval.metadata["source_log"]
            for sample in saved.samples:
                events = monitor_events(sample)
                assert len(events) == 1
                event = events[0]
                assert f"CASE={sample.id}" in event.input[-1].text
                assert f"Checked {sample.id}" in event.output.completion
                assert event.cache == expected_cache
                assert not event.pending and event.error is None
                expected = 0.9 if sample.metadata["eval_mode"] == "attack" else 0.1
                assert sample.scores["max_suspicion_scorer"].value == expected
                assert CALLS[sample.id] == 1


@pytest.mark.parametrize("wrapped", [False, True])
def test_live_sample_cancellation_retains_request(tmp_path, wrapped):
    sources = make_source_logs(tmp_path / "source", count=1)
    source = sources[0].samples[0]
    SLOW.add(source.id)
    protocol = trusted_monitoring_builder(
        policy=submit_policy(), monitor=make_monitor()
    )
    if wrapped:
        protocol = with_model_error_logging(wrapped=protocol)

    @solver
    def run_protocol():
        async def solve(state, generate):
            result = await protocol(AgentState(messages=state.messages), [])
            state.messages = result.messages
            return state

        return solve

    try:
        task = Task(
            name="cancelled-monitor",
            dataset=[
                Sample(id=source.id, input=source.input, metadata=source.metadata)
            ],
            solver=run_protocol(),
            time_limit=1,
            model="none/none",
        )
        result = inspect_eval(task, log_dir=str(tmp_path / "live"), display="none")[0]
    finally:
        SLOW.remove(source.id)
    saved = read_eval_log(result.location, resolve_attachments="full")
    sample = saved.samples[0]
    event = monitor_events(sample)[0]
    assert any(e.event == "sample_limit" for e in sample.events)
    assert f"CASE={sample.id}" in event.input[-1].text
    assert not event.output.completion, "Cancellation must not invent a response"
    assert event.output.usage is None
    if wrapped:
        assert not event.pending
        assert event.completed is not None
        assert "interrupted" in event.error
    else:
        assert (
            event.pending is True
        ), "Baseline no longer reproduces; review compatibility shim"


def test_parallel_monitor_cancellation_does_not_change_sibling(tmp_path):
    """One cancelled invocation must not finalize another invocation's request."""
    from inspect_ai.log import transcript
    from control_arena.eval import run_monitor_for_messages

    sources = make_source_logs(tmp_path / "source", count=1)
    source = sources[0].samples[0]
    SLOW.add(source.id)

    @solver
    def run_pair():
        async def solve(state, generate):
            async def call():
                return await run_monitor_for_messages(
                    with_model_error_logging(wrapped=make_monitor()),
                    state.messages,
                    metadata=state.metadata,
                )

            first = asyncio.create_task(call())
            second = asyncio.create_task(call())
            try:
                for _ in range(100):
                    events = [e for e in transcript().events if e.event == "model"]
                    if len(events) == 2:
                        break
                    await asyncio.sleep(0.01)
                assert len(events) == 2
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)
                assert sum(bool(e.pending) for e in events) == 1
                assert sum(bool(e.error) for e in events) == 1
            finally:
                first.cancel()
                second.cancel()
                await asyncio.gather(first, second, return_exceptions=True)
            state.completed = True
            return state

        return solve

    try:
        task = Task(
            name="parallel-monitor",
            dataset=[
                Sample(id=source.id, input=source.messages, metadata=source.metadata)
            ],
            solver=run_pair(),
            model="none/none",
        )
        log = inspect_eval(task, log_dir=str(tmp_path / "parallel"), display="none")[0]
        assert log.status == "success", log.error
        assert log.samples[0].error is None
    finally:
        SLOW.remove(source.id)
