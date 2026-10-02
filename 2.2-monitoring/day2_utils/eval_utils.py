"""Shared utilities for loading and processing ControlArena evaluation logs."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from inspect_ai.log import EvalLog, EvalSample, list_eval_logs, read_eval_log, write_eval_log

# Reasons ControlArena's monitor records when a reply has no usable score. It then
# falls back to a suspicion score of 0, which must not count as "looks safe".
UNSCORED_REASONS = {
    "No score found in model response",
    "Could not parse score from model response",
}


def has_unscored_monitor_reply(sample: EvalSample) -> bool:
    """True if any monitor reply in this sample had no parseable score."""
    return any(
        message.role == "assistant"
        and (message.metadata or {}).get("reason") in UNSCORED_REASONS
        for message in sample.messages
    )


def unscored_sample_ids(logs: Sequence[EvalLog], eval_mode: str) -> set:
    """IDs of samples in ``eval_mode`` whose monitor score is a parse-failure 0."""
    return {
        sample.id
        for log in logs
        for sample in log.samples or []
        if sample.metadata.get("eval_mode") == eval_mode
        and has_unscored_monitor_reply(sample)
    }


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
    return [read_eval_log(info, resolve_attachments="full") for info in log_infos]


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

    Samples whose monitor reply had no parseable score are skipped: their
    stored score is a fallback 0, not a judgement.

    Returns (honest_scores, attack_scores) as numpy arrays.
    """
    honest: list[float] = []
    attack: list[float] = []
    for log in scored_logs:
        for sample in log.samples or []:
            score_val = sample.scores.get("max_suspicion_scorer")
            if score_val is None or score_val.value is None:
                continue
            if has_unscored_monitor_reply(sample):
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


def show_monitor_replies(
    scored_logs: list[EvalLog],
    eval_mode: str | None = None,
    failed_only: bool = False,
    sample_id: str | int | None = None,
    n: int = 3,
    max_chars: int = 3000,
) -> None:
    """Print the monitor's raw replies from monitoring eval logs.

    The same information is in ``inspect view``: open the log under
    ``logs/*_scored/`` and choose the sample's **Transcript** tab.

    Args:
        scored_logs: Logs returned by ``run_monitor_evaluation`` (or the ``logs``
            attribute of an ``IncompleteMonitorEvaluation``).
        eval_mode: Only show "honest" or "attack" samples.
        failed_only: Only show samples without a valid score (timeouts,
            provider errors, malformed or truncated replies).
        sample_id: Only show this APPS problem.
        n: Maximum number of samples to print.
        max_chars: Truncate each reply to this many characters.
    """
    shown = 0
    for log in scored_logs:
        for sample in log.samples or []:
            score = (sample.scores or {}).get("max_suspicion_scorer")
            failed = sample.error is not None or score is None
            if eval_mode is not None and sample.metadata.get("eval_mode") != eval_mode:
                continue
            if failed_only and not failed:
                continue
            if sample_id is not None and str(sample.id) != str(sample_id):
                continue
            if shown >= n:
                return
            shown += 1

            status = "FAILED: " + sample.error.message.splitlines()[0] if sample.error else f"score {score.value}"
            print(f"\n=== sample {sample.id} ({sample.metadata.get('eval_mode')}) | {status}")
            for event in sample.events:
                if event.event != "model":
                    continue
                if event.error:
                    print(f"[model call error: {event.error}]")
                    continue
                output = event.output
                tokens = output.usage.output_tokens if output.usage else "?"
                cut = ", hit max_tokens" if output.stop_reason == "max_tokens" else ""
                print(f"[{tokens} output tokens{cut}]")
                reply = output.message.text
                print(reply if len(reply) <= max_chars else reply[:max_chars] + " [...]")
