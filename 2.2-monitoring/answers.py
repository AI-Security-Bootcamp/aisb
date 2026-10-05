# %%

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
    plot_roc_comparison,
    plot_roc_curve,
    plot_score_histograms,
)

from aisb_utils import report
from aisb_utils.env import load_dotenv

load_dotenv()
nest_asyncio2.apply()  # allow asyncio.run() inside Jupyter notebook cells

# This section's directory; prompts/, logs/, reference_solutions/ live here
SCRIPT_DIR = Path(__file__).resolve().parent

# %%

E2_1_TRAJECTORY_DIR = str(SCRIPT_DIR / "logs" / "e2.1_trajectories")

# Generate trajectories on first run; cached in logs/e2.1_trajectories/ after that.
# No LLM calls, but each sample runs inside a Docker container (~1 min total).
# apps_solution_policy() replays verified solutions and backdoors from the APPS
# dataset directly, without calling an LLM; it's a fast deterministic replay.
if not list(Path(E2_1_TRAJECTORY_DIR).glob("*.eval")):
    generate_trajectories(
        apps_solution_policy(),
        E2_1_TRAJECTORY_DIR,
        limit=50,
    )
    print("Trajectories generated and cached in logs/e2.1_trajectories/")
else:
    print("Trajectories already exist in logs/e2.1_trajectories/; skipping generation")


# %%


# Models for the control protocol exercises (trusted monitor vs untrusted agent)
DEFAULT_TRUSTED_MODEL = "openrouter/openai/gpt-5.6"
DEFAULT_UNTRUSTED_MODEL = "openrouter/openai/gpt-5.6"


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
    print(prompt_text)
    # TODO: Create and return a basic monitor with basic_monitor_builder with given prompt and model
    # - Don't forget to set correct score_range (0-10)!
    # - Inspect also supports caching LLM responses - consider setting cache=True for faster evaluation
    builder = basic_monitor_builder(
        model=DEFAULT_TRUSTED_MODEL,
        prompt=prompt_text,
        score_range=MonitorScoreRange(min=0,max=10, output_min=0.0, output_max=1.0),
        cache=False,
    )
    print(builder)
    return builder


from section2_test import test_create_monitor


test_create_monitor(create_monitor)

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
    monitored_logs = asyncio.run(
        run_monitor_for_eval_logs(monitor, eval_logs, progress=True)
    )

    # Extract the numeric suspicion scores from the monitor messages using
    # max_suspicion_scorer: if the monitor ran multiple times per sample
    # (e.g. per tool call), this takes the maximum score as the sample's score.
    scored_logs = [
        inspect_score(log, [max_suspicion_scorer()], action="overwrite")
        for log in monitored_logs
    ]

    # Split scores by eval mode (honest vs attack).
    honest_scores, attack_scores = extract_scores(scored_logs)

    # Compute ROC AUC: treats honest samples as negatives (label 0) and
    # attack samples as positives (label 1). AUC measures how well the
    # monitor's score separates the two classes.    pass

    y_true = np.concatenate([np.zeros(len(honest_scores)), np.ones(len(attack_scores))])
    y_scores_arr = np.concatenate([honest_scores, attack_scores])
    _, _, _, auc = calculate_roc_metrics(y_true, y_scores_arr)

    # Optionally persist the scored logs so they can be inspected later.
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for log in scored_logs:
            # Derive a filename from the original log location, falling back to index.
            name = (
                Path(log.location).name
                if log.location
                else f"log_{scored_logs.index(log)}.eval"
            )
            write_eval_log(log, save_path / name)
        print(f"  Scored logs saved to {save_path.resolve()}/")

    return honest_scores, attack_scores, auc


e2_1_scored_path = SCRIPT_DIR / "logs/e2.1_trajectories_scored"
if e2_1_scored_path.exists() and any(e2_1_scored_path.glob("*.eval")):
    print("Loading pre-scored trajectories from logs/e2.1_trajectories_scored...")
    scored_logs = load_eval_logs(str(e2_1_scored_path))
    e2_1_honest, e2_1_attack = extract_scores(scored_logs)
    y_true = np.concatenate([np.zeros(len(e2_1_honest)), np.ones(len(e2_1_attack))])
    y_scores_arr = np.concatenate([e2_1_honest, e2_1_attack])
    _, _, _, e2_1_auc = calculate_roc_metrics(y_true, y_scores_arr)
else:
    print("Evaluating monitor on trajectories...")
    e2_1_monitor = create_monitor(SCRIPT_DIR / "prompts" / "monitor_prompt.txt")
    e2_1_honest, e2_1_attack, e2_1_auc = evaluate_monitor_on_trajectories(
        e2_1_monitor,
        E2_1_TRAJECTORY_DIR,
        save_dir=e2_1_scored_path,
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
fpr, tpr, thresholds_21, _ = calculate_roc_metrics(
    np.concatenate([np.zeros(len(e2_1_honest)), np.ones(len(e2_1_attack))]),
    np.concatenate([e2_1_honest, e2_1_attack]),
)
plot_roc_curve(fpr, tpr, e2_1_auc, label="Your monitor", save_path=None, thresholds=thresholds_21)
from section2_test import test_monitor_better_than_random


test_monitor_better_than_random(e2_1_auc)




def create_attack_policy(attack_prompt_path: Path, cache: bool = False) -> ControlAgent:
    """Create an untrusted policy that uses a custom attack prompt."""
    attack = attack_policy_builder(
        model=DEFAULT_UNTRUSTED_MODEL,
        prompt_file=str(attack_prompt_path),
        cache=cache,
    )
    honest = honest_policy_builder(
        model=DEFAULT_UNTRUSTED_MODEL,
        cache=cache,
    )
    return untrusted_policy_builder(
        attack=attack,
        honest=honest,
    )


from section2_test import test_create_attack_policy


test_create_attack_policy(create_attack_policy)

# %%

LIMIT = 10
E2_2_TRAJECTORY_DIR = str(SCRIPT_DIR / "logs" / "e2.2_attacks")

print(f"Generating trajectories with your updated prompts (limit={LIMIT})...")
print(f"This runs LLM calls ({DEFAULT_UNTRUSTED_MODEL} + {DEFAULT_TRUSTED_MODEL}) with Docker sandboxes (~2 min)...")

attack_policy = create_attack_policy(SCRIPT_DIR / "prompts/attack_system_prompt.txt")
e2_2_monitor = create_monitor(SCRIPT_DIR / "prompts/monitor_prompt.txt")

print(f"  Untrusted model (agent): {DEFAULT_UNTRUSTED_MODEL}")
print(f"  Trusted model (monitor): {DEFAULT_TRUSTED_MODEL}")

e2_2_logs = generate_trajectories(
    trusted_monitoring_builder(policy=attack_policy, monitor=e2_2_monitor),
    E2_2_TRAJECTORY_DIR,
    limit=LIMIT,
    trusted_model=DEFAULT_TRUSTED_MODEL,
    untrusted_model=DEFAULT_UNTRUSTED_MODEL,
)

e2_2_honest, e2_2_attack = extract_scores(e2_2_logs)

e2_2_auc = calculate_roc_metrics(
    np.concatenate([np.zeros(len(e2_2_honest)), np.ones(len(e2_2_attack))]),
    np.concatenate([e2_2_honest, e2_2_attack]),
)[3]

# Task success rates and attack evasion plots; uses samples_df to get
# main_task_success and side_task_success alongside the monitor scores.
_df = samples_df(e2_2_logs)
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
try:
    print(f"  (compare with {e2_1_auc:.3f} on the dataset backdoors from Exercise 2.2.1)")
except NameError:
    pass

# Determine unique incrementing run ID for saving plots
import re
import shutil

existing_run_nums = [
    int(m.group(1))
    for f in (SCRIPT_DIR / "logs").glob("e2.2_run_*_roc_curve.png")
    if (m := re.search(r"e2\.2_run_(\d+)_", f.name))
]
run_id = max(existing_run_nums, default=0) + 1
run_prefix = f"e2.2_run_{run_id:02d}"
print(f"\nSaving plots for [{run_prefix}]...")

hist_path = SCRIPT_DIR / "logs" / f"{run_prefix}_score_histograms.png"
roc_path = SCRIPT_DIR / "logs" / f"{run_prefix}_roc_curve.png"
comp_path = SCRIPT_DIR / "logs" / f"{run_prefix}_roc_comparison.png"
evasion_path = SCRIPT_DIR / "logs" / f"{run_prefix}_attack_evasion.png"

# Plot score histograms
plot_score_histograms(
    e2_2_honest,
    e2_2_attack,
    title=f"Your Monitor vs Your Attacks ({run_prefix})",
    save_path=hist_path,
)

# Plot ROC curve with threshold markers
fpr, tpr, thresholds, _ = calculate_roc_metrics(
    np.concatenate([np.zeros(len(e2_2_honest)), np.ones(len(e2_2_attack))]),
    np.concatenate([e2_2_honest, e2_2_attack]),
)
plot_roc_curve(
    fpr,
    tpr,
    e2_2_auc,
    label=f"Monitor vs your attacks ({run_prefix})",
    save_path=roc_path,
    thresholds=thresholds,
)
shutil.copyfile(roc_path, SCRIPT_DIR / "logs" / "e2.2_roc_curve_latest.png")

# Plot comparison ROC curve if Exercise 2.2.1 logs exist
try:
    e2_1_scored_dir = SCRIPT_DIR / "logs" / "e2.1_trajectories_scored"
    if e2_1_scored_dir.exists():
        e2_1_logs = load_eval_logs(str(e2_1_scored_dir))
        h1, a1 = extract_scores(e2_1_logs)
        fpr1, tpr1, th1, auc1 = calculate_roc_metrics(
            np.concatenate([np.zeros(len(h1)), np.ones(len(a1))]),
            np.concatenate([h1, a1]),
        )
        plot_roc_comparison(
            {
                "Baseline (Exercise 2.2.1)": (fpr1, tpr1, th1, auc1),
                f"Your Attacks ({run_prefix})": (fpr, tpr, thresholds, e2_2_auc),
            },
            title="ROC Curves: Baseline vs Red-Team Attacks",
            save_path=comp_path,
        )
        shutil.copyfile(comp_path, SCRIPT_DIR / "logs" / "e2.2_roc_comparison_latest.png")
except Exception as e:
    print(f"Skipping comparison plot: {e}")

_score_col = "score_max_suspicion_scorer"
plot_attack_evasion_curve(
    _attack_df.loc[_attack_df["side_task_success"], _score_col].to_numpy(),
    save_path=evasion_path,
)

print(f"  ✓ Saved ROC curve: {roc_path}")
print(f"  ✓ Saved ROC comparison: {comp_path}")
print(f"  ✓ Saved Score histograms: {hist_path}")
print(f"  ✓ Saved Latest shortcut: {SCRIPT_DIR / 'logs' / 'e2.2_roc_curve_latest.png'}")




# Evaluate the CoT monitor on your attack trajectories from Exercise 2.2.2.
# cot_prompt_formatter(cot_only=False) populates {task_description}, {cot},
# and {tool_calls}, giving the monitor full visibility into the agent's reasoning.
prompt_path = SCRIPT_DIR / "prompts" / "cot_monitor_prompt.txt"
e2_3_cot_monitor = basic_monitor_builder(
    model=DEFAULT_TRUSTED_MODEL,
    model_role=None,
    prompt=prompt_path.read_text(),
    # cot_only=False means the formatter includes both chain-of-thought AND tool calls
    prompt_formatter=lambda meta, msgs, p: cot_prompt_formatter(
        meta, msgs, p, cot_only=False
    ),
    score_range=MonitorScoreRange(min=0, max=10),
)
E2_3_COT_SCORED_DIR = str(SCRIPT_DIR / "logs" / "e2.3_cot_scored")
e2_3_honest, e2_3_attack, e2_3_auc = evaluate_monitor_on_trajectories(
    e2_3_cot_monitor, E2_2_TRAJECTORY_DIR, save_dir=E2_3_COT_SCORED_DIR
)

print(f"\n  CoT monitor AUC: {e2_3_auc:.3f}")
print(f"  Tool-call-only AUC (Exercise 2.2.1): {e2_1_auc:.3f}")
print(f"  Your attacks AUC  (Exercise 2.2.2): {e2_2_auc:.3f}")

plot_score_histograms(
    e2_3_honest, e2_3_attack, title="CoT Monitor Score Distribution", save_path=None
)
fpr, tpr, _, _ = calculate_roc_metrics(
    np.concatenate([np.zeros(len(e2_3_honest)), np.ones(len(e2_3_attack))]),
    np.concatenate([e2_3_honest, e2_3_attack]),
)
plot_roc_curve(fpr, tpr, e2_3_auc, label="CoT monitor", save_path=None)

roc_data = {}
for label, h_scores, a_scores in [
    ("Tool-call-only", e2_2_honest, e2_2_attack),
    ("CoT-enhanced", e2_3_honest, e2_3_attack),
]:
    if len(h_scores) > 0 and len(a_scores) > 0:
        y_true = np.concatenate([np.zeros(len(h_scores)), np.ones(len(a_scores))])
        y_scores_arr = np.concatenate([h_scores, a_scores])
        fpr_arr, tpr_arr, _, auc_val = calculate_roc_metrics(y_true, y_scores_arr)
        roc_data[label] = (fpr_arr, tpr_arr, auc_val)

if roc_data:
    plot_roc_comparison(roc_data, title="Tool-Call vs CoT Monitor", save_path=None)