import asyncio
import sys
from pathlib import Path

import nest_asyncio
import numpy as np
from control_arena.eval import run_monitor_for_eval_logs
from control_arena.scorers import max_suspicion_scorer
from inspect_ai import score as inspect_score
from inspect_ai.log import write_eval_log

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
# day2_utils lives alongside this section (prompts/, logs/, reference_solutions/ too).
for _path in [str(_root), str(Path(__file__).resolve().parent)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
from control_arena.monitor import MonitorScoreRange, basic_monitor_builder
from day2_utils.eval_utils import extract_scores, load_eval_logs
from day2_utils.metrics import calculate_roc_metrics
from day2_utils.plotting import (
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
