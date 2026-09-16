"""Run the course ROC path on real trajectories and verify saved resumption."""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import nest_asyncio2
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "issue_38"))
from reproduce_apps import SECTION, audit_log, load_course_functions
from day2_utils.metrics import calculate_roc_metrics
from day2_utils.monitoring import IncompleteMonitorEvaluation
from day2_utils.plotting import plot_roc_curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--passes", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    load_dotenv(args.env_file)
    nest_asyncio2.apply()
    helpers = load_course_functions()
    monitor = helpers["create_monitor"](
        SECTION / "reference_solutions/monitor_prompt.txt"
    )
    evaluate = helpers["evaluate_monitor_on_trajectories"]
    original_runner = evaluate.__globals__["run_monitor_evaluation"]
    latest_logs = []

    def observed_runner(*positional, **keywords):
        nonlocal latest_logs
        try:
            latest_logs = original_runner(*positional, **keywords)
        except IncompleteMonitorEvaluation as error:
            latest_logs = error.logs
            raise
        return latest_logs

    evaluate.__globals__["run_monitor_evaluation"] = observed_runner
    results = []
    started = time.monotonic()
    for attempt in range(1, args.passes + 1):
        try:
            honest, attack, auc = evaluate(
                monitor, str(args.input), args.output / "scored"
            )
        except IncompleteMonitorEvaluation as error:
            result = {"pass": attempt, "complete": False, "coverage": error.coverage}
        else:
            result = {
                "pass": attempt,
                "complete": True,
                "honest": len(honest),
                "attack": len(attack),
                "auc": float(auc),
            }
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["logs"] = [audit_log(log.location) for log in latest_logs]
        results.append(result)
        (args.output / "passes.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps({key: value for key, value in result.items() if key != "logs"}),
            flush=True,
        )
        if result["complete"]:
            break
    else:
        raise RuntimeError(
            "Monitor remains incomplete; saved failures need investigation"
        )

    fpr, tpr, _, _ = calculate_roc_metrics(
        np.concatenate([np.zeros(len(honest)), np.ones(len(attack))]),
        np.concatenate([honest, attack]),
    )
    plot_roc_curve(fpr, tpr, auc, save_path=args.output / "roc.png")
    previous_paths = sorted(log.location for log in latest_logs)
    repeated = evaluate(monitor, str(args.input), args.output / "scored")
    assert sorted(log.location for log in latest_logs) == previous_paths
    assert np.array_equal(repeated[0], honest) and np.array_equal(repeated[1], attack)
    assert repeated[2] == auc
    summary = {
        "complete": True,
        "passes": len(results),
        "honest": len(honest),
        "attack": len(attack),
        "auc": float(auc),
        "completed_rerun_reused_same_eval_logs": True,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
