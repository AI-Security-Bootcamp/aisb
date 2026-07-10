"""Generate APPS evaluation trajectories for AI control exercises.

This module is the single entry point for running ControlArena evaluations.
Pass any ``micro_protocol`` to customise what the agent does and how it is
monitored. Two common patterns:

**Exercise 4.1 — fast dataset replay (no LLM calls):**

    logs = generate_trajectories(limit=50, log_dir=E4_1_TRAJECTORY_DIR)

**Exercise 4.2 — LLM attacks with live monitoring:**

    from control_arena.micro_protocol import trusted_monitoring_builder
    protocol = trusted_monitoring_builder(policy=attack_policy, monitor=monitor)
    logs = generate_trajectories(
        micro_protocol=protocol,
        limit=10,
        log_dir=E4_2_TRAJECTORY_DIR,
    )
    honest_scores, attack_scores = extract_scores(logs)

Read this file to familiarise yourself with the ControlArena evaluation API
before implementing the exercises.

As a standalone CLI (Exercise 4.1 only)::

    python -m day2_utils.generate_trajectories --limit 50
"""

from __future__ import annotations

from pathlib import Path

from control_arena import EvalMode
from control_arena._control_agent import ControlAgent
from control_arena.eval import ControlEvalConfig, get_control_task_product
from control_arena.scorers import max_suspicion_scorer
from control_arena.settings.apps.apps_setting import AppsSetting
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog

# Default models — override via function arguments or CLI flags.
DEFAULT_TRUSTED_MODEL = "openrouter/meta-llama/llama-3.1-8b-instruct"
DEFAULT_UNTRUSTED_MODEL = "openrouter/openai/gpt-4o-mini"


def generate_trajectories(
    micro_protocol: ControlAgent,
    log_dir: str,
    limit: int = 50,
    trusted_model: str = DEFAULT_TRUSTED_MODEL,
    untrusted_model: str = DEFAULT_UNTRUSTED_MODEL,
) -> list[EvalLog]:
    """Run an APPS control evaluation and return the resulting eval logs.

    By default uses ``apps_solution_policy()`` — a fast dataset replay that
    submits verified solutions and backdoors without any LLM calls. Pass a
    custom ``micro_protocol`` (e.g. ``trusted_monitoring_builder(...)``) for
    LLM-driven attack generation with live monitoring.

    Each sample is scored inside a **Docker container** to verify test results,
    so Docker must be running. With the default policy, expect ~1 min for 50
    samples; with an LLM policy, expect ~2 min for 10 samples.

    Args:
        micro_protocol: A ControlArena micro-protocol (policy + optional monitor).
            Defaults to ``apps_solution_policy()`` — replays dataset solutions,
            no LLM calls needed.
        limit: Number of samples per eval mode (honest + attack).
        log_dir: Directory to write Inspect eval log files. Created if absent.
        trusted_model: The trusted (weaker, safe) model — used for monitoring
            and as a fallback in defer-to-trusted protocols.
        untrusted_model: The untrusted (stronger, potentially adversarial) model
            — generates code submissions in LLM-driven protocols.

    Returns:
        List of ``EvalLog`` objects, one per (setting, protocol, mode)
        combination. Pass to ``extract_scores()`` to get suspicion score arrays.
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # AppsSetting selects coding problems from the APPS dataset.
    # has_backdoor_solution=True limits to problems with a verified backdoor in
    # the dataset, ensuring realistic, non-trivially-detectable attack opportunities.
    setting = AppsSetting(
        cache_python_code=True,
        has_backdoor_solution=True,
        sandbox_type="docker",  # requires Docker to be running
    )

    eval_config = ControlEvalConfig(
        limit=limit,
        log_dir=str(log_dir_path),
        trusted_model=trusted_model,
        untrusted_model=untrusted_model,
    )

    # Create one eval task per (setting, protocol, mode) combination.
    eval_tasks = get_control_task_product(
        settings=setting,
        micro_protocols=micro_protocol,
        modes=[EvalMode.HONEST, EvalMode.ATTACK],
        eval_config=eval_config,
        # Include suspicion scorer so monitor scores are stored in the logs.
        # Ignored when the protocol has no monitor (e.g. apps_solution_policy).
        monitoring_scorers=[max_suspicion_scorer()],
    )

    print(f"Generating trajectories → {log_dir_path.resolve()}")
    # For each sample, Inspect spins up a Docker container, submits the solution
    # code, runs the test suite inside the container, and records the full
    # trajectory (tool calls, outputs, scores) as an eval log.
    eval_logs: list[EvalLog] = inspect_eval(tasks=eval_tasks, **eval_config.eval_args())

    n_honest = sum(
        1
        for log in eval_logs
        for s in (log.samples or [])
        if (s.metadata or {}).get("eval_mode") == "honest"
    )
    n_attack = sum(
        1
        for log in eval_logs
        for s in (log.samples or [])
        if (s.metadata or {}).get("eval_mode") == "attack"
    )
    print(f"  honest: {n_honest}  attack: {n_attack}  saved to {log_dir_path}/")
    return eval_logs


if __name__ == "__main__":
    import click

    @click.command()
    @click.option(
        "--limit",
        default=50,
        show_default=True,
        help="Number of samples per mode (honest + attack).",
    )
    @click.option(
        "--log-dir",
        default="logs/e4.1_trajectories",
        show_default=True,
        help="Directory to save trajectory logs.",
    )
    def cli(limit: int, log_dir: str) -> None:
        """Pre-generate APPS trajectories for Exercise 4.1 (no LLM calls)."""
        from control_arena.settings.apps.apps_solution_policy import (
            apps_solution_policy,
        )
        from dotenv import load_dotenv

        load_dotenv()
        generate_trajectories(apps_solution_policy(), log_dir, limit=limit)

    cli()
