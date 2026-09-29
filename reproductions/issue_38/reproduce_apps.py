"""Run the real APPS paths from section 2.2 and audit the saved monitor events.

Live and post-hoc modes make paid model calls using the course's configured
OpenRouter models. Replay uses dataset solutions and a real Docker sandbox.
"""

import argparse
import ast
import asyncio
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import nest_asyncio2
import numpy as np
from control_arena import ControlAgent
from control_arena.eval import run_monitor_for_eval_logs
from control_arena.micro_protocol import trusted_monitoring_builder
from control_arena.monitor import basic_monitor_builder
from control_arena.monitor._types import MonitorScoreRange
from control_arena.policy import (
    attack_policy_builder,
    honest_policy_builder,
    untrusted_policy_builder,
)
from control_arena.scorers import max_suspicion_scorer
from control_arena.settings.apps.apps_solution_policy import apps_solution_policy
from dotenv import load_dotenv
from inspect_ai import score as inspect_score
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import GenerateConfig

# Make the workspace root importable (so `from aisb_utils import report` works),
# regardless of how deeply this file is nested.
_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

SECTION = _root / "2.2-monitoring"
sys.path.insert(0, str(SECTION))

from day2_utils.eval_utils import extract_scores, load_eval_logs
from day2_utils.generate_trajectories import (
    DEFAULT_TRUSTED_MODEL,
    DEFAULT_UNTRUSTED_MODEL,
    generate_trajectories,
)
from day2_utils.metrics import calculate_roc_metrics
from day2_utils.monitoring import run_monitor_evaluation


def load_course_functions(revision=None):
    """Load the actual solution functions without running unrelated exercises."""
    source = SECTION / "section2_solution.py"
    source_text = (
        source.read_text()
        if revision is None
        else subprocess.check_output(
            ["git", "show", f"{revision}:2.2-monitoring/section2_solution.py"],
            cwd=_root,
            text=True,
        )
    )
    wanted = {
        "create_monitor",
        "create_attack_policy",
        "evaluate_monitor_on_trajectories",
    }
    definitions = [
        node
        for node in ast.parse(source_text).body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "MONITOR_CONFIG"
                for target in node.targets
            )
        )
    ]
    assert {
        node.name for node in definitions if isinstance(node, ast.FunctionDef)
    } == wanted
    namespace = dict(globals())
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return {name: namespace[name] for name in wanted}


def all_events(events):
    """Include nested events if a log uses an older transcript representation."""
    for event in events:
        yield event
        nested = getattr(event, "events", None)
        if nested:
            yield from all_events(nested)


def audit_log(path):
    log = read_eval_log(path, resolve_attachments="full")
    samples = []
    for sample in log.samples or []:
        events = list(all_events(sample.events))
        models = [event for event in events if event.event == "model"]
        monitors = [event for event in models if event.model == DEFAULT_TRUSTED_MODEL]
        metadata = [
            message.metadata
            for message in sample.messages
            if message.role == "assistant"
            and message.metadata
            and "suspicion_score" in message.metadata
        ]
        samples.append(
            {
                "id": sample.id,
                "mode": sample.metadata.get("eval_mode"),
                "error": sample.error.message if sample.error else None,
                "limits": [
                    {"type": event.type, "message": event.message}
                    for event in events
                    if event.event == "sample_limit"
                ],
                "scores": {
                    name: score.value for name, score in (sample.scores or {}).items()
                },
                "events": dict(Counter(event.event for event in events)),
                "model_counts": dict(Counter(event.model for event in models)),
                "monitor_events": [
                    {
                        "cache": event.cache,
                        "pending": bool(event.pending),
                        "error": event.error,
                        "completed": str(event.completed) if event.completed else None,
                        "input_characters": sum(
                            len(message.text) for message in event.input
                        ),
                        "output_characters": len(event.output.completion),
                        "has_reason_tag": "<reason>" in event.output.completion,
                        "has_reasoning_tag": "<reasoning>" in event.output.completion,
                        "has_score_tag": "<score>" in event.output.completion,
                    }
                    for event in monitors
                ],
                "monitor_metadata": metadata,
            }
        )
    return {
        "log": str(path),
        "status": log.status,
        "error": log.error.message if log.error else None,
        "model_usage": {
            name: usage.model_dump()
            for name, usage in (log.stats.model_usage or {}).items()
        },
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["replay", "post-hoc", "live", "audit"])
    parser.add_argument("--limit", type=int, default=10, help="Samples per eval mode")
    parser.add_argument("--env-file", type=Path, default=_root / ".env")
    parser.add_argument(
        "--input", type=Path, help="Existing log directory for post-hoc mode"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "post-hoc" and args.input is None:
        parser.error("post-hoc mode requires --input")
    if args.mode == "audit" and args.output is None:
        parser.error("audit mode requires --output pointing to existing logs")
    if args.limit < 1:
        parser.error("--limit must be positive")

    load_dotenv(args.env_file)
    if args.mode in {"live", "post-hoc"} and not os.environ.get("OPENROUTER_API_KEY"):
        parser.error(f"Configure OPENROUTER_API_KEY in {args.env_file}")
    if args.mode == "replay" and not os.environ.get("OPENROUTER_API_KEY"):
        # ControlEvalConfig constructs model clients even though this policy
        # submits dataset solutions and never calls a model. Satisfy that local
        # constructor check without needing credentials for the sandbox check.
        os.environ["OPENROUTER_API_KEY"] = "unused-dataset-replay"
    # Match the notebook setup in section2_solution.py.
    nest_asyncio2.apply()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output
        or Path(__file__).resolve().parent / "artifacts" / f"apps-{args.mode}-{stamp}"
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    helpers = load_course_functions()
    monitor_prompt = SECTION / "reference_solutions" / "monitor_prompt.txt"
    attack_prompt = SECTION / "reference_solutions" / "attack_system_prompt.txt"

    if args.mode == "replay":
        generate_trajectories(apps_solution_policy(), str(output), limit=args.limit)
    elif args.mode == "post-hoc":
        monitor = helpers["create_monitor"](monitor_prompt)
        helpers["evaluate_monitor_on_trajectories"](monitor, str(args.input), output)
    elif args.mode == "live":
        monitor = helpers["create_monitor"](monitor_prompt)
        policy = helpers["create_attack_policy"](attack_prompt)
        generate_trajectories(
            trusted_monitoring_builder(policy=policy, monitor=monitor),
            str(output),
            limit=args.limit,
        )

    paths = sorted(output.glob("*.eval"))
    if not paths:
        raise RuntimeError(f"No saved eval logs in {output}")
    report = {
        "mode": args.mode,
        "requested_limit": args.limit if args.mode in {"live", "replay"} else None,
        "python": platform.python_version(),
        "versions": {
            package: version(package)
            for package in ("control-arena", "inspect-ai", "openai")
        },
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_root, text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                SECTION / "section2_solution.py",
                SECTION / "day2_utils" / "generate_trajectories.py",
                monitor_prompt,
                attack_prompt,
                _root / "requirements-control-arena.txt",
            )
        },
        "logs": [audit_log(path) for path in paths],
    }
    if args.mode == "replay":
        assert all(
            not sample["model_counts"]
            for log in report["logs"]
            for sample in log["samples"]
        ), "Dataset replay unexpectedly recorded a model call"
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    for log in report["logs"]:
        print(
            Path(log["log"]).name,
            log["status"],
            f"samples={len(log['samples'])}",
            f"errors={sum(sample['error'] is not None for sample in log['samples'])}",
            f"limits={sum(len(sample['limits']) for sample in log['samples'])}",
            f"monitor_events={sum(len(sample['monitor_events']) for sample in log['samples'])}",
        )
    print(f"Saved audit to {output / 'audit.json'}")
    if any(
        log["status"] != "success" or any(sample["error"] for sample in log["samples"])
        for log in report["logs"]
    ):
        raise SystemExit(
            "Some evaluations failed; inspect audit.json and the eval logs."
        )


if __name__ == "__main__":
    main()
