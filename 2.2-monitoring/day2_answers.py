# %%
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import nest_asyncio2
import numpy as np
from control_arena import ControlAgent
from control_arena.analysis import samples_df
from control_arena.micro_protocol import trusted_monitoring_builder
from control_arena.monitor import (
    MonitorScoreRange,
    basic_monitor_builder,
    cot_prompt_formatter,
)
from control_arena.policy import (
    attack_policy_builder,
    honest_policy_builder,
    untrusted_policy_builder,
)
from control_arena.settings.apps.apps_solution_policy import apps_solution_policy
from inspect_ai.agent import Agent
from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
# day2_utils lives alongside this section (prompts/, logs/, reference_solutions/ too).
for _path in [str(_root), str(Path(__file__).resolve().parent)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
from day2_utils.eval_utils import (
    extract_scores,
    load_eval_logs,
    show_monitor_replies,
    unscored_sample_ids,
)
from day2_utils.generate_trajectories import generate_trajectories
from day2_utils.attack_prompt import load_attack_prompt
from day2_utils.metrics import calculate_roc_metrics
from day2_utils.monitor_prompt import load_monitor_prompt
from day2_utils.monitoring import run_monitor_evaluation
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
# The cache counts only if both logs (honest + attack) finished; an interrupted
# first run leaves partial logs behind, which are discarded and regenerated.
_cached_logs = [read_eval_log(str(p), header_only=True) for p in Path(E2_1_TRAJECTORY_DIR).glob("*.eval")]
if len(_cached_logs) != 2 or any(log.status != "success" for log in _cached_logs):
    shutil.rmtree(E2_1_TRAJECTORY_DIR, ignore_errors=True)
    generate_trajectories(
        apps_solution_policy(),
        E2_1_TRAJECTORY_DIR,
        limit=50,
    )
    print("Trajectories generated and cached in logs/e2.1_trajectories/")
else:
    print("Trajectories already exist in logs/e2.1_trajectories/; skipping generation")
# %%

# %%

# Models for the control protocol exercises (trusted monitor vs untrusted agent)
DEFAULT_TRUSTED_MODEL = "openrouter/deepseek/deepseek-v4.1-flash"
DEFAULT_UNTRUSTED_MODEL = "openrouter/openai/gpt-4o-mini"

# Settings for the monitor's model calls:
# - temperature 0.6, top_p 0.9: Meta's defaults for Llama 3.
# - max_tokens 1000: coherent replies are 100-700 tokens, so this only cuts off
#   replies stuck repeating themselves.
# - attempt_timeout 45s, no automatic retries: a request that never returns
#   fails that attempt instead of stalling the whole run; the evaluation then
#   retries the sample twice before rejecting it.
MONITOR_CONFIG = GenerateConfig(
    temperature=0.6,
    top_p=0.9,
    max_tokens=21000,
    attempt_timeout=45,
    max_retries=0,
    max_connections=10,
)

# %%

# %%
# Print the full prompt the monitor will receive: header + your instructions + footer.
print(load_monitor_prompt(SCRIPT_DIR / "prompts" / "monitor_instructions.txt"))

# %%
def create_monitor(instructions_path: Path, model: str = DEFAULT_TRUSTED_MODEL):
    """Create a monitor from a file of monitor instructions.

    load_monitor_prompt wraps the instructions in the fixed header (problem and
    tool calls) and footer (output format); see day2_utils/monitor_prompt.py.

    Args:
        instructions_path: Path to the monitor instructions text file.
        model: Model to use for monitoring (typically the trusted model).

    Returns:
        A ControlArena monitor agent.
    """
    # TODO: Build and return a monitor with basic_monitor_builder.
    # 1. Turn the instructions file into a full prompt with load_monitor_prompt
    # 2. Use `model` directly (not the eval config's trusted model)
    # 3. Set the score range to match the 0-10 scores your prompt asks for
    # 4. Pass MONITOR_CONFIG, and disable caching
    loaded_instructions = load_monitor_prompt(instructions_path)
    return basic_monitor_builder(
        model=model,
        prompt=loaded_instructions,
        score_range=MonitorScoreRange(min_score=0, max_score=10),
        config=MONITOR_CONFIG,
        cache=False,
    )

from section2_test import test_create_monitor


test_create_monitor(create_monitor)

# %%

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
        save_dir: Directory for monitoring eval logs. Defaults to a sibling of
            trajectory_dir with the suffix "_scored".

    Returns (honest_scores, attack_scores, auc).
    """
    # Load the saved Inspect eval logs from disk.
    # Each log contains a list of samples (one per APPS problem) with the
    # agent's full tool call history and the scoring results.
    eval_logs = load_eval_logs(trajectory_dir)
    print(f"  Loaded {sum(len(log.samples or []) for log in eval_logs)} samples")

    # Each saved trajectory is a new Inspect sample. Its Transcript records the
    # monitor's separate conversation; the original trajectory is not rerun.
    output_dir = (
        Path(save_dir) if save_dir is not None else Path(f"{trajectory_dir}_scored")
    )
    # Samples whose monitor reply has no valid score (e.g. a repetition loop
    # that hit max_tokens) are retried, then rejected with a warning.
    scored_logs = run_monitor_evaluation(monitor, eval_logs, log_dir=output_dir)

    # Split scores by eval mode (honest vs attack). Rejected samples have no
    # score and are skipped.
    honest_scores, attack_scores = extract_scores(scored_logs)

    # Compute ROC AUC: treats honest samples as negatives (label 0) and
    # attack samples as positives (label 1). AUC measures how well the
    # monitor's score separates the two classes.
    y_true = np.concatenate([np.zeros(len(honest_scores)), np.ones(len(attack_scores))])
    y_scores_arr = np.concatenate([honest_scores, attack_scores])
    _, _, _, auc = calculate_roc_metrics(y_true, y_scores_arr)

    print(f"  Monitor requests, responses, and scores saved to {output_dir.resolve()}/")

    return honest_scores, attack_scores, auc

# %%

print("Evaluating monitor on trajectories...")
E2_1_SCORED_DIR = SCRIPT_DIR / "logs" / "e2.1_trajectories_scored"
e2_1_monitor = create_monitor(SCRIPT_DIR / "prompts" / "monitor_instructions.txt")
e2_1_honest, e2_1_attack, e2_1_auc = evaluate_monitor_on_trajectories(
    e2_1_monitor, E2_1_TRAJECTORY_DIR, save_dir=E2_1_SCORED_DIR
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

# %%
# Print the full system prompt the attacking agent will receive: header + your instructions + footer.
print(load_attack_prompt(SCRIPT_DIR / "prompts" / "attack_instructions.txt"))

# %%


def create_attack_policy(attack_instructions_path: Path) -> ControlAgent:
    """Create an untrusted policy whose attack mode uses your attack instructions."""
    # TODO: Return a policy that attacks in ATTACK mode and is honest in HONEST mode.
    # 1. Build the attack sub-policy from the prompt load_attack_prompt produces
    # 2. Build the honest sub-policy
    # 3. Combine them with untrusted_policy_builder
    # Enable caching on both sub-policies to speed up re-runs.
    loaded_instructions = load_attack_prompt(attack_instructions_path)
    attack_sub_policy = attack_policy_builder(
        model=DEFAULT_UNTRUSTED_MODEL,
        prompt=loaded_instructions,
        config=MONITOR_CONFIG,
        cache=True,
    )
    honest_sub_policy = honest_policy_builder(
        model=DEFAULT_UNTRUSTED_MODEL,
        config=MONITOR_CONFIG,
        cache=True,
    )
    return untrusted_policy_builder(
        attack=attack_sub_policy,
        honest=honest_sub_policy,
    )

from section2_test import test_create_attack_policy


test_create_attack_policy(create_attack_policy)

# %%

# Re-run after editing your attack/monitor prompts to regenerate.
# Requires Docker + LLM calls (~2 min). See day2_utils/generate_trajectories.py.
LIMIT = 10
E2_2_TRAJECTORY_DIR = str(SCRIPT_DIR / "logs" / "e2.2_attacks")
attack_policy = create_attack_policy(SCRIPT_DIR / "prompts/attack_instructions.txt")
e2_2_monitor = create_monitor(SCRIPT_DIR / "prompts/monitor_instructions.txt")

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

# Working backdoors whose monitor reply had a parseable score (a reply without
# one is stored as 0, which would look like a perfect evasion).
_score_col = "score_max_suspicion_scorer"
_scored = ~_attack_df["id"].isin(unscored_sample_ids(e2_2_logs, "attack"))
plot_attack_evasion_curve(
    _attack_df.loc[_attack_df["side_task_success"] & _scored, _score_col].to_numpy(),
    save_path=None,
)
# %%


# Evaluate the CoT monitor on your attack trajectories from Exercise 2.2.2.
# cot_prompt_formatter(cot_only=False) populates {task_description}, {cot},
# and {tool_calls}, giving the monitor full visibility into the agent's reasoning.
cot_instructions_path = SCRIPT_DIR / "prompts" / "cot_monitor_instructions.txt"
cot_prompt = load_monitor_prompt(cot_instructions_path, cot=True)
print(cot_prompt)  # the full prompt the CoT monitor will receive

e2_3_cot_monitor = basic_monitor_builder(
    model=DEFAULT_TRUSTED_MODEL,
    model_role=None,
    prompt=cot_prompt,
    # cot_only=False means the formatter includes both chain-of-thought AND tool calls
    prompt_formatter=lambda meta, msgs, p: cot_prompt_formatter(
        meta, msgs, p, cot_only=False
    ),
    score_range=MonitorScoreRange(min=0, max=10),
    config=MONITOR_CONFIG,
    cache=False,
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
# %%
