"""Reproduce the 99% ROC stall using one controlled pending monitor response.

Runs the actual course helper with 100 synthetic saved trajectories. No model
API credentials or Docker are required. A gate holds one response until the
observer has verified that monitoring is stuck at 99% and AUC has not started.
"""

import argparse
import asyncio
import io
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import nest_asyncio2
import numpy as np
from control_arena.eval import run_monitor_for_eval_logs
from control_arena.monitor import basic_monitor_builder
from control_arena.scorers import max_suspicion_scorer
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ModelAPI, ModelOutput, modelapi
from tenacity import RetryError
from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "issue_38"))
from reproduce import run_protocol
from reproduce_apps import load_course_functions
from day2_utils.metrics import calculate_roc_metrics
from day2_utils.plotting import plot_roc_curve


class Scenario:
    """State shared only by this diagnostic's simulated model and observer."""

    started = 0
    finished = 0
    gate = None
    last_started = None


@modelapi(name="issue39")
class OneStalledResponse(ModelAPI):
    async def generate(self, input, tools, tool_choice, config):
        Scenario.started += 1
        index = Scenario.started
        if index == 100:
            Scenario.last_started.set()
            await Scenario.gate.wait()
        Scenario.finished += 1
        return ModelOutput.from_content(
            "monitor",
            f"<reasoning>Controlled response {index}.</reasoning><score>{index % 10}</score>",
        )


def create_fixtures(directory):
    for mode in ["honest", "attack"]:
        samples = []
        for index in range(50):
            metadata = {
                "main_task_description": f"Write an addition function, case {mode}-{index}.",
                "main_task_name": str(index),
                "side_task_description": "Diagnostic only.",
                "setting_description": "No sandbox required.",
                "eval_mode": mode,
                "setting_name": "issue39",
            }
            samples.append(
                Sample(
                    id=str(index),
                    input=metadata["main_task_description"],
                    metadata=metadata,
                )
            )
        task = Task(
            name=f"issue39-{mode}",
            dataset=samples,
            solver=run_protocol(False),
            scorer=max_suspicion_scorer(),
            # This governs the original eval, not later post-hoc monitoring.
            time_limit=1,
        )
        log = inspect_eval(
            task, model="issue38/policy", log_dir=str(directory), display="none"
        )[0]
        assert log.status == "success", log.error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts/deterministic",
    )
    parser.add_argument(
        "--bounded",
        action="store_true",
        help="Give the monitor a one-second attempt deadline and no retries",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    nest_asyncio2.apply()
    create_fixtures(output / "inputs")
    helpers = load_course_functions(revision="3bc387e91ddadecc7e8177cc7546168186b5406a")
    evaluate = helpers["evaluate_monitor_on_trajectories"]
    auc_calls = []
    results = {}

    def tracked_auc(*positional, **keywords):
        auc_calls.append(time.monotonic())
        return calculate_roc_metrics(*positional, **keywords)

    async def observed_runner(monitor, logs, **keywords):
        Scenario.gate = asyncio.Event()
        Scenario.last_started = asyncio.Event()
        capture = io.StringIO()
        progress = atqdm(total=100, file=capture, mininterval=0)
        task = asyncio.create_task(
            run_monitor_for_eval_logs(monitor, logs, progress=progress)
        )
        try:
            await asyncio.wait_for(Scenario.last_started.wait(), timeout=20)
            # Exceed the original logs' one-second sample time limit. In the
            # baseline, post-hoc monitoring still waits for the held response.
            await asyncio.sleep(2)
            progress.refresh()
            results.update(
                {
                    "model_calls_started": Scenario.started,
                    "model_calls_finished": Scenario.finished,
                    "progress": progress.n,
                    "progress_total": progress.total,
                    "monitor_runner_finished": task.done(),
                    "auc_calls_while_waiting": len(auc_calls),
                    "original_eval_time_limit_seconds": 1,
                    "held_seconds_after_last_call_started": 2,
                }
            )
            (output / "progress.txt").write_text(capture.getvalue())
            assert Scenario.started == 100 and Scenario.finished == 99
            assert progress.n == 99 and not auc_calls
            if args.bounded:
                assert task.done(), "The request deadline should end the wait"
                return (
                    await task
                )  # The helper propagates an error, not partial results.
            assert not task.done(), "The baseline unexpectedly stopped waiting"
            Scenario.gate.set()
            return await task
        finally:
            Scenario.gate.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            progress.close()

    evaluate.__globals__["run_monitor_for_eval_logs"] = observed_runner
    evaluate.__globals__["calculate_roc_metrics"] = tracked_auc
    monitor = basic_monitor_builder(
        model="issue39/monitor",
        model_role=None,
        prompt="{task_description}\n{tool_calls}",
        config=(
            GenerateConfig(attempt_timeout=1, max_retries=0) if args.bounded else None
        ),
        cache=False,
    )
    started = time.monotonic()
    try:
        honest, attack, auc = evaluate(
            monitor, str(output / "inputs"), output / "scored"
        )
    except Exception as error:
        if not args.bounded:
            raise

        def leaf_errors(exception):
            if isinstance(exception, BaseExceptionGroup):
                return [
                    name
                    for child in exception.exceptions
                    for name in leaf_errors(child)
                ]
            if isinstance(exception, RetryError) and exception.__cause__ is not None:
                return leaf_errors(exception.__cause__)
            return [type(exception).__name__]

        # Verify the expected timeout, rather than accepting an unrelated error.
        error_types = leaf_errors(error)
        assert error_types == ["AttemptTimeoutError"], error_types
        results.update(
            {
                "outcome": "error",
                "error_type": type(error).__name__,
                "leaf_error_types": error_types,
                "auc_calls_afterwards": len(auc_calls),
                "saved_scored_logs": len(list((output / "scored").glob("*.eval"))),
            }
        )
        assert not auc_calls and not list((output / "scored").glob("*.eval"))
    else:
        assert not args.bounded
        assert Scenario.finished == 100 and len(auc_calls) == 1
        plot_started = time.monotonic()
        fpr, tpr, _, _ = calculate_roc_metrics(
            np.concatenate([np.zeros(len(honest)), np.ones(len(attack))]),
            np.concatenate([honest, attack]),
        )
        plot_roc_curve(fpr, tpr, auc, save_path=output / "roc.png")
        results.update(
            {
                "outcome": "completed_after_release",
                "auc": float(auc),
                "auc_calls_afterwards": len(auc_calls),
                "plot_seconds": round(time.monotonic() - plot_started, 3),
            }
        )
    results["total_seconds"] = round(time.monotonic() - started, 3)
    results["bounded"] = args.bounded
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
