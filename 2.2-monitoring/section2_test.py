# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
import nest_asyncio2
import numpy as np
from control_arena import ControlAgent
from control_arena.analysis import samples_df
from control_arena.eval import run_monitor_for_eval_logs
from control_arena.micro_protocol import trusted_monitoring_builder
from control_arena.monitor import (
    basic_monitor_builder,
    cot_prompt_formatter,
)
from control_arena.monitor._types import MonitorScoreRange
from control_arena.policy import (
    attack_policy_builder,
    honest_policy_builder,
    untrusted_policy_builder,
)
from control_arena.scorers import max_suspicion_scorer
from control_arena.settings.apps.apps_solution_policy import apps_solution_policy
from inspect_ai import score as inspect_score
from inspect_ai.agent import Agent
from inspect_ai.log import write_eval_log
from day2_utils.eval_utils import extract_scores, load_eval_logs
from day2_utils.generate_trajectories import generate_trajectories
from day2_utils.metrics import calculate_roc_metrics
from day2_utils.plotting import (
    plot_attack_evasion_curve,
    plot_roc_comparison,
    plot_roc_curve,
    plot_score_histograms,
)
from aisb_utils import report
from aisb_utils.env import load_dotenv



@report
def test_create_monitor(solution: Callable[..., object]):
    # requires: ControlArena installed (builds a monitor Agent; no LLM call needed)
    monitor = solution(Path(__file__).parent / "prompts" / "monitor_prompt.txt")
    assert isinstance(monitor, Agent), f"Expected an Agent, got {type(monitor)}"
    print("  Monitor created successfully!")




@report
def test_monitor_better_than_random(auc: float):
    """Check that the monitor performs better than random guessing."""
    # NOTE: unlike the other tests, this takes a precomputed `auc: float`
    # (the result of evaluating the monitor), not a `solution` callable.
    # requires: the AUC to have been produced by a live monitor run
    # (Docker + ControlArena + OpenRouter key).
    assert auc > 0.5, (
        f"Monitor AUC ({auc:.3f}) should be > 0.5 (random). Improve your prompt!"
    )
    print(f"  AUC = {auc:.3f} (better than random)")




@report
def test_create_attack_policy(solution: Callable[[Path], object]):
    # requires: ControlArena installed (builds a policy ControlAgent; no LLM call needed)
    policy = solution(Path(__file__).parent / "prompts" / "attack_system_prompt.txt")
    assert isinstance(policy, ControlAgent), (
        f"create_attack_policy() should return a ControlAgent, got {type(policy)}"
    )
    print("  create_attack_policy() returns a ControlAgent ✓")
