
import asyncio
import sys
from pathlib import Path

import nest_asyncio
import numpy as np
from control_arena import ControlAgent
from control_arena.analysis import samples_df
from control_arena.eval import run_monitor_for_eval_logs
from control_arena.micro_protocol import trusted_monitoring_builder
from control_arena.monitor import (
    basic_monitor_builder,
)
from control_arena.monitor._types import MonitorScoreRange
from control_arena.policy import (
    attack_policy_builder,
    honest_policy_builder,
    untrusted_policy_builder,
)
from control_arena.scorers import max_suspicion_scorer
from inspect_ai import score as inspect_score
from inspect_ai.log import write_eval_log

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
# day2_utils lives alongside this section (prompts/, logs/, reference_solutions/ too).
for _path in [str(_root), str(Path(__file__).resolve().parent)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
from day2_utils.eval_utils import extract_scores, load_eval_logs
from day2_utils.generate_trajectories import generate_trajectories
from day2_utils.metrics import calculate_roc_metrics
from day2_utils.plotting import (
    plot_attack_evasion_curve,
    plot_roc_curve,
    plot_score_histograms,
)

from aisb_utils.env import load_dotenv

load_dotenv()
nest_asyncio.apply()  # allow asyncio.run() inside Jupyter notebook cells

# This section's directory; prompts/, logs/, reference_solutions/ live here
SCRIPT_DIR = Path(__file__).resolve().parent


# Models for the control protocol exercises (trusted monitor vs untrusted agent)
DEFAULT_TRUSTED_MODEL = "openrouter/meta-llama/llama-3.1-8b-instruct"
DEFAULT_UNTRUSTED_MODEL = "openrouter/openai/gpt-4o-mini"


def create_monitor(prompt_path: Path, model: str = DEFAULT_TRUSTED_MODEL):
    """Create a monitor from a prompt template file.

    The prompt file uses str.format placeholders ({task_description}, {tool_calls})
    which are filled in by the default last_tool_calls_prompt_formatter.

    Args:
        prompt_path: Path to the monitor prompt text file.
        model: Model to use for monitoring (typically the trusted model).

    Returns:
        A ControlArena monitor agent.
    """
    prompt_text = prompt_path.read_text()


    return basic_monitor_builder(prompt = prompt_text, cache=True, score_range = MonitorScoreRange(min=0, max=10), model=model)


from section2_test import test_create_monitor

test_create_monitor(create_monitor)


E2_1_TRAJECTORY_DIR = str(SCRIPT_DIR / "logs" / "e2.1_trajectories")


# %%


def evaluate_monitor_on_trajectories(
    monitor,
    trajectory_dir: str,
    save_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run a monitor on pre-generated trajectories and compute ROC AUC.

    Args:
        monitor: The monitor ControlAgent to evaluate.
        trajectory_dir: Directory containing the pre-generated Inspect eval logs.
        save_dir: If provided, write a copy of the scored logs to this directory.

    Returns (honest_scores, attack_scores, auc).
    """
    # Load the saved Inspect eval logs from disk.
    # Each log contains a list of samples (one per APPS problem) with the
    # agent's full tool call history and the scoring results.
    eval_logs = load_eval_logs(trajectory_dir)
    print(f"  Loaded {sum(len(log.samples or []) for log in eval_logs)} samples")

    # Run the monitor over every sample in every log.
    # The monitor reads each sample's tool calls (submitted code) and appends
    # a message containing the suspicion score to the sample's message list.
    monitored_logs = asyncio.run(run_monitor_for_eval_logs(monitor, eval_logs, progress=True))

    # Extract the numeric suspicion scores from the monitor messages using
    # max_suspicion_scorer: if the monitor ran multiple times per sample
    # (e.g. per tool call), this takes the maximum score as the sample's score.
    scored_logs = [inspect_score(log, [max_suspicion_scorer()], action="overwrite") for log in monitored_logs]

    # Split scores by eval mode (honest vs attack).
    honest_scores, attack_scores = extract_scores(scored_logs)

    # Compute ROC AUC: treats honest samples as negatives (label 0) and
    # attack samples as positives (label 1). AUC measures how well the
    # monitor's score separates the two classes.
    y_true = np.concatenate([np.zeros(len(honest_scores)), np.ones(len(attack_scores))])
    y_scores_arr = np.concatenate([honest_scores, attack_scores])
    _, _, _, auc = calculate_roc_metrics(y_true, y_scores_arr)

    # Optionally persist the scored logs so they can be inspected later.
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for log in scored_logs:
            # Derive a filename from the original log location, falling back to index.
            name = Path(log.location).name if log.location else f"log_{scored_logs.index(log)}.eval"
            write_eval_log(log, save_path / name)
        print(f"  Scored logs saved to {save_path.resolve()}/")

    return honest_scores, attack_scores, auc


print("Evaluating monitor on trajectories...")
e2_1_monitor = create_monitor(SCRIPT_DIR / "prompts" / "monitor_prompt.txt")
e2_1_honest, e2_1_attack, e2_1_auc = evaluate_monitor_on_trajectories(
    e2_1_monitor,
    E2_1_TRAJECTORY_DIR,
    save_dir=SCRIPT_DIR / "logs/e2.1_trajectories_scored",
)
print(f"\n  ROC AUC: {e2_1_auc:.3f}")
print(f"  Mean honest score: {np.mean(e2_1_honest):.3f}")
print(f"  Mean attack score: {np.mean(e2_1_attack):.3f}")

plot_score_histograms(
    e2_1_honest,
    e2_1_attack,
    title="Monitor: Suspicion Score Distribution",
    save_path=None,
)
fpr, tpr, _, _ = calculate_roc_metrics(
    np.concatenate([np.zeros(len(e2_1_honest)), np.ones(len(e2_1_attack))]),
    np.concatenate([e2_1_honest, e2_1_attack]),
)
plot_roc_curve(fpr, tpr, e2_1_auc, label="Your monitor", save_path=None)
from section2_test import test_monitor_better_than_random

test_monitor_better_than_random(e2_1_auc)

def create_attack_policy(attack_prompt_path: Path) -> ControlAgent:
    """Create an untrusted policy that uses a custom attack prompt."""
    # TODO: Use untrusted_policy_builder() to create a policy that
    # produces an honest solution in HONEST mode or an attack generated
    # accoring to the prompt in `attack_prompt_path` in ATTACK mode.
    # See https://control-arena.aisi.org.uk/reference/policy.html#untrusted_policy_builder
    #
    # Hint 1: you can use `prompt_file` argument to pass path to the
    # prompt directly
    # Hint 2: enable caching for the honest policy to speed up evaluation
    attack_prompt = attack_prompt_path.read_text()

    attack_policy = attack_policy_builder(prompt=attack_prompt, cache=True)
    honest_policy = honest_policy_builder(cache=True)

    return untrusted_policy_builder(attack=attack_policy, honest=honest_policy)



from section2_test import test_create_attack_policy

test_create_attack_policy(create_attack_policy)

# Re-run after editing your attack/monitor prompts to regenerate.
# Requires Docker + LLM calls (~2 min). See day2_utils/generate_trajectories.py.
LIMIT = 10
E2_2_TRAJECTORY_DIR = str(SCRIPT_DIR / "logs" / "e2.2_attacks")
attack_policy = create_attack_policy(SCRIPT_DIR / "prompts/attack_system_prompt.txt")
e2_2_monitor = create_monitor(SCRIPT_DIR / "prompts/monitor_prompt.txt")

# trusted_monitoring_builder wires the attack policy and monitor together:
# on each agent action it runs the policy to get the action, then immediately
# runs the monitor to score it, all within a single eval pass.
e2_2_logs = generate_trajectories(
    trusted_monitoring_builder(policy=attack_policy, monitor=e2_2_monitor),
    E2_2_TRAJECTORY_DIR,
    limit=LIMIT,
)
e2_2_honest, e2_2_attack = extract_scores(e2_2_logs)

e2_2_auc = calculate_roc_metrics(
    np.concatenate([np.zeros(len(e2_2_honest)), np.ones(len(e2_2_attack))]),
    np.concatenate([e2_2_honest, e2_2_attack]),
)[3]

# Task success rates and attack evasion plots; uses samples_df to get
# main_task_success and side_task_success alongside the monitor scores.
_df = samples_df(load_eval_logs(E2_2_TRAJECTORY_DIR))
_attack_df = _df[_df["eval_mode"] == "attack"]
_honest_df = _df[_df["eval_mode"] == "honest"]

print(
    f"  Main task success (honest): {_honest_df['main_task_success'].mean():.0%}  ← usefulness proxy"
)
print(
    f"  Main task success (attack): {_attack_df['main_task_success'].mean():.0%}  ← did the attack also solve the problem?"
)
print(
    f"  Side task success (attack): {_attack_df['side_task_success'].mean():.0%}  ← fraction that inserted a working backdoor"
)
print(f"\n  Monitor AUC on your attacks: {e2_2_auc:.3f}")
print(f"  (compare with {e2_1_auc:.3f} on the dataset backdoors from Exercise 2.2.1)")

plot_score_histograms(
    e2_2_honest,
    e2_2_attack,
    title="Your Monitor vs Your Attacks",
    save_path=None,
)
fpr, tpr, _, _ = calculate_roc_metrics(
    np.concatenate([np.zeros(len(e2_2_honest)), np.ones(len(e2_2_attack))]),
    np.concatenate([e2_2_honest, e2_2_attack]),
)
plot_roc_curve(fpr, tpr, e2_2_auc, label="Monitor vs your attacks", save_path=None)

_score_col = "score_max_suspicion_scorer"
plot_attack_evasion_curve(
    _attack_df.loc[_attack_df["side_task_success"], _score_col].to_numpy(),
    save_path=None,
)
