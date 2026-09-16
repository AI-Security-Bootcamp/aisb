"""Trace the unchanged section 2.2 post-hoc monitor/ROC path with real models."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import nest_asyncio2
import numpy as np
from control_arena import control_agent, control_task_metadata
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "issue_38"))
from reproduce_apps import SECTION, load_course_functions
from day2_utils.metrics import calculate_roc_metrics
from day2_utils.plotting import plot_roc_curve, plot_score_histograms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--env-file", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    load_dotenv(args.env_file)
    nest_asyncio2.apply()
    started = time.monotonic()
    trace_path = output / "trace.jsonl"

    def record(event, **fields):
        entry = {
            "event": event,
            "seconds": round(time.monotonic() - started, 3),
            "utc": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        with trace_path.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")

    helpers = load_course_functions()
    inner = helpers["create_monitor"](
        SECTION / "reference_solutions/monitor_prompt.txt"
    )

    @control_agent
    def traced_monitor():
        async def impl(state, tools):
            metadata = control_task_metadata()
            identity = {
                "mode": metadata.eval_mode.value,
                "sample": metadata.main_task_name,
            }
            record("monitor_started", **identity)
            call_started = time.monotonic()
            try:
                result = await inner(state, tools)
            except BaseException as error:
                record("monitor_failed", error_type=type(error).__name__, **identity)
                raise
            monitor_metadata = result.messages[-1].metadata or {}
            record(
                "monitor_finished",
                duration=round(time.monotonic() - call_started, 3),
                score=monitor_metadata.get("suspicion_score"),
                reason=monitor_metadata.get("reason"),
                usage=monitor_metadata.get("model_usage"),
                **identity,
            )
            return result

        return impl

    evaluate = helpers["evaluate_monitor_on_trajectories"]
    original_runner = evaluate.__globals__["run_monitor_for_eval_logs"]

    async def traced_runner(*positional, **keywords):
        record("monitor_phase_started")
        result = await original_runner(*positional, **keywords)
        record("monitor_phase_finished")
        return result

    # Add observation around the actual helper without replacing its monitor,
    # concurrency, cache, scoring, timeout settings, or plot implementation.
    evaluate.__globals__["run_monitor_for_eval_logs"] = traced_runner
    record("run_started")
    try:
        honest, attack, auc = evaluate(
            traced_monitor(), str(args.input), output / "scored"
        )
        record("scoring_and_auc_finished", auc=float(auc))
        plot_started = time.monotonic()
        plot_score_histograms(honest, attack, save_path=output / "histogram.png")
        fpr, tpr, _, _ = calculate_roc_metrics(
            np.concatenate([np.zeros(len(honest)), np.ones(len(attack))]),
            np.concatenate([honest, attack]),
        )
        plot_roc_curve(fpr, tpr, auc, save_path=output / "roc.png")
        record("plots_finished", duration=round(time.monotonic() - plot_started, 3))
    except BaseException as error:
        record("run_failed", error_type=type(error).__name__)
        raise
    record(
        "run_finished",
        honest_samples=len(honest),
        attack_samples=len(attack),
        auc=float(auc),
    )


if __name__ == "__main__":
    main()
