"""Shared utilities for loading and processing ControlArena evaluation logs."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log, write_eval_log


def load_eval_logs(log_dir: str, latest_only: bool = True) -> list[EvalLog]:
    """Load eval logs from a directory.

    Args:
        log_dir: Directory containing .eval log files.
        latest_only: If True, load only the most recent honest and attack logs
            (determined by the timestamp prefix in the filename).

    Raises FileNotFoundError if no logs are found.
    """
    log_infos = list_eval_logs(log_dir)  # sorted descending (newest first)
    if not log_infos:
        raise FileNotFoundError(
            f"No eval logs found in {log_dir}. "
            "Run the corresponding generate_trajectories.py first."
        )
    if latest_only:
        # Keep the first (most recent) file whose name contains "honest"
        # and the first whose name contains "attack".
        selected = []
        found_honest = found_attack = False
        for info in log_infos:
            name = Path(info.name).name
            if not found_honest and "honest" in name:
                selected.append(info)
                found_honest = True
            elif not found_attack and "attack" in name:
                selected.append(info)
                found_attack = True
            if found_honest and found_attack:
                break
        log_infos = selected
    return [read_eval_log(info) for info in log_infos]


def save_eval_logs(
    logs: Sequence[EvalLog], output_dir: Path, suffix: str = ""
) -> list[Path]:
    """Write eval logs to an output directory as .eval files.

    Returns the list of paths written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, log in enumerate(logs):
        filename = f"log_{i}{suffix}.eval"
        path = output_dir / filename
        write_eval_log(log, location=path)
        paths.append(path)
    return paths


def extract_scores(scored_logs: list[EvalLog]) -> tuple[np.ndarray, np.ndarray]:
    """Extract honest and attack suspicion scores from scored eval logs.

    Returns (honest_scores, attack_scores) as numpy arrays.
    """
    honest: list[float] = []
    attack: list[float] = []
    for log in scored_logs:
        for sample in log.samples or []:
            score_val = sample.scores.get("max_suspicion_scorer")
            if score_val is None or score_val.value is None:
                continue
            try:
                parsed = float(score_val.value)
            except (ValueError, TypeError):
                parsed = float("nan")
            mode = sample.metadata.get("eval_mode", "")
            if mode == "honest":
                honest.append(parsed)
            elif mode == "attack":
                attack.append(parsed)
    return np.array(honest), np.array(attack)
