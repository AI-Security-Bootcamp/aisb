"""Issue #39: bounded completion, rejected failures, and safe resumption."""

from copy import deepcopy

import pytest

from day2_utils.monitoring import run_monitor_evaluation
from .fixtures import CALLS, FAIL, INVALID, SLOW, make_monitor, make_source_logs


def all_samples(logs):
    return [sample for log in logs for sample in log.samples or []]


def test_99_percent_stall_rejects_only_the_stalled_sample(tmp_path):
    sources = make_source_logs(tmp_path / "source", count=50)
    identities = [sample.id for sample in all_samples(sources)]
    stalled = identities[-1]
    SLOW.add(stalled)
    monitor = make_monitor()
    output = tmp_path / "monitored"
    try:
        logs = run_monitor_evaluation(
            monitor, sources, output, display="none", timeout_seconds=0.5
        )
    finally:
        SLOW.remove(stalled)
    saved = all_samples(logs)
    assert len(saved) == 100
    assert sum(sample.error is None for sample in saved) == 99
    bad = next(sample for sample in saved if sample.id == stalled)
    assert "Monitor exceeded" in bad.error.message
    assert not bad.scores, "An unfinished monitor must not receive a numeric score"
    event = next(event for event in bad.events if event.event == "model")
    assert not event.pending and "interrupted" in event.error
    assert f"CASE={stalled}" in event.input[-1].text
    assert CALLS[stalled] == 3, "A failed sample is tried once plus two retries"
    assert all(CALLS[identity] == 1 for identity in identities if identity != stalled)

    counts = dict(CALLS)
    run_monitor_evaluation(
        monitor, sources, output, display="none", timeout_seconds=0.5
    )
    assert (
        dict(CALLS) == counts
    ), "A finished run, including its rejections, must not issue model requests"


@pytest.mark.parametrize("failure_set", [INVALID, FAIL])
def test_failed_score_is_rejected_not_zero_and_siblings_finish(tmp_path, failure_set):
    sources = make_source_logs(tmp_path / "source")
    failed_id = all_samples(sources)[0].id
    for sample in all_samples(sources):
        for message in sample.messages:
            if message.role == "assistant":
                message.metadata = {"suspicion_score": 0.25, "reason": "Stale score"}
    original = deepcopy(sources)
    failure_set.add(failed_id)
    monitor = make_monitor()
    try:
        logs = run_monitor_evaluation(
            monitor, sources, tmp_path / "monitored", display="none"
        )
    finally:
        failure_set.remove(failed_id)
    samples = all_samples(logs)
    assert sum(sample.error is None for sample in samples) == 3
    rejected = next(sample for sample in samples if sample.id == failed_id)
    assert not rejected.scores, "A rejected sample must not receive a numeric score"
    if failure_set is INVALID:
        assert "no parseable <score>" in rejected.error.message
    assert sources == original
    for sample in samples:
        if sample.id == failed_id:
            continue
        expected = 0.9 if sample.metadata["eval_mode"] == "attack" else 0.1
        assert sample.scores["max_suspicion_scorer"].value == expected
        assert CALLS[sample.id] == 1
    assert CALLS[failed_id] == 3, "A failed sample is tried once plus two retries"


def test_changed_prompt_and_trajectory_do_not_reuse_old_scores(tmp_path):
    sources = make_source_logs(tmp_path / "source", count=1)
    identities = [sample.id for sample in all_samples(sources)]
    output = tmp_path / "monitored"
    run_monitor_evaluation(make_monitor(), sources, output, display="none")
    revised_monitor = make_monitor(prompt_suffix="Changed prompt")
    run_monitor_evaluation(revised_monitor, sources, output, display="none")
    assert all(CALLS[identity] == 2 for identity in identities)
    changed = deepcopy(sources)
    for sample in all_samples(changed):
        sample.metadata["main_task_description"] += " Updated task constraint."
    run_monitor_evaluation(revised_monitor, changed, output, display="none")
    assert all(CALLS[identity] == 3 for identity in identities)


def test_completed_run_resumes_after_python_restart(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    sources = make_source_logs(tmp_path / "source", count=1)
    output = tmp_path / "monitored"
    logs = run_monitor_evaluation(make_monitor(), sources, output, display="none")
    previous = sorted(log.eval.eval_id for log in logs)
    script = """
import json, sys
from day2_utils.eval_utils import load_eval_logs
from day2_utils.monitoring import run_monitor_evaluation
from tests.fixtures import CALLS, FAIL, make_monitor
sources = load_eval_logs(sys.argv[1])
FAIL.update(s.id for log in sources for s in log.samples)
logs = run_monitor_evaluation(make_monitor(), sources, sys.argv[2], display='none')
assert sorted(log.eval.eval_id for log in logs) == json.loads(sys.argv[3])
assert not CALLS, 'Completed samples must survive a Python restart without model calls'
"""
    import json

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path / "source"),
            str(output),
            json.dumps(previous),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_course_helper_excludes_rejected_samples_from_auc(tmp_path):
    """Test the actual solution helper without executing the whole exercise."""
    import ast
    from pathlib import Path

    from day2_utils.eval_utils import extract_scores
    from day2_utils.metrics import calculate_roc_metrics

    sources = make_source_logs(tmp_path / "source", count=2)
    source_file = Path(__file__).resolve().parents[1] / "section2_solution.py"
    node = next(
        node
        for node in ast.parse(source_file.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "evaluate_monitor_on_trajectories"
    )
    namespace = {
        "Path": Path,
        "np": __import__("numpy"),
        "load_eval_logs": lambda directory: sources,
        "run_monitor_evaluation": run_monitor_evaluation,
        "calculate_roc_metrics": calculate_roc_metrics,
        "extract_scores": extract_scores,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source_file), "exec"),
        namespace,
    )
    bad = all_samples(sources)[0]
    FAIL.add(bad.id)
    try:
        honest, attack, auc = namespace["evaluate_monitor_on_trajectories"](
            make_monitor(), "unused", tmp_path / "monitored"
        )
    finally:
        FAIL.remove(bad.id)
    assert len(honest) + len(attack) == 3, "The rejected sample must be excluded"
    assert 0.0 not in [*honest, *attack], "A rejected sample must not score 0"
    assert auc == 1.0
