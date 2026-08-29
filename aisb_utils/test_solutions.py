#!/usr/bin/env python3
"""
Run every section's reference solution end-to-end and report what breaks.

This is a *verification* tool for content owners, not part of the participant
build. `build_instructions.py` turns a `*_solution.py` into the participant
handout; this script turns the same file into the filled-in answer -- every
`if "SOLUTION":` branch taken, every `test_*` call left in place -- executes it,
and tells you which sections fail.

Workflow for one section, e.g. 1.2-logprobs/section2_solution.py:

    1. Generate 1.2-logprobs/section2_answers.py containing the correct code.
    2. Run it with the section folder as the working directory, so its
       `from section2_test import test_...` imports resolve.
    3. Delete the answers file immediately, pass or fail.

Step 3 is the default and should stay that way on any machine participants can
reach: the answers file contains every solution for the section. It is written,
executed, and removed inside a try/finally, a stale-file sweep at startup cleans
up after a hard kill, and `*_answers.py` is gitignored so an interrupted run can
never be committed.

Pass --remain to keep the generated files instead, so a failing section can be
edited and re-run by hand. They are still swept away at the start of the next
run, and still gitignored -- but on a shared pod they are readable answers
sitting in the repo until then.

Usage:
    python aisb_utils/test_solutions.py --day 3        # every 3.x section
    python aisb_utils/test_solutions.py --day 1 --day 2
    python aisb_utils/test_solutions.py --all          # every section
    python aisb_utils/test_solutions.py --all --list   # dry run, list only
    python aisb_utils/test_solutions.py --day 1 --remain   # keep them to debug

Exit status is 0 only if every section ran to completion with exit code 0.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# This script lives at <repo>/aisb_utils/test_solutions.py. Resolve the repo
# root from __file__ rather than the working directory so the script can be
# invoked from anywhere (the deploy script runs it over SSH from $HOME).
AISB_UTILS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AISB_UTILS_DIR.parent

# solution_parsing.py is a sibling, imported by module name the same way
# build_instructions.py does it.
if str(AISB_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(AISB_UTILS_DIR))

from solution_parsing import build_reference_py  # noqa: E402

SOLUTION_SUFFIX = "_solution.py"
ANSWERS_SUFFIX = "_answers.py"
TEST_SUFFIX = "_test.py"

# Content folders are named "<day>.<section>-slug" (1.2-logprobs), except Day 0
# which predates the numbered layout.
CONTENT_DIR_RE = re.compile(r"^\d+\.\d+-")
DAY0_DIR_NAME = "day0-setup"

# Sections that train models or download weights legitimately take a long time;
# the timeout only exists to stop a hung section from blocking the whole run.
DEFAULT_TIMEOUT = 1800

# How much of a failing section's output to replay in the final summary. The
# full output has already been streamed live; this is just so the summary is
# self-contained.
FAILURE_TAIL_LINES = 40


# ─────────────────────────────────────────────────────────────────────────────
# Discovering which sections to run
# ─────────────────────────────────────────────────────────────────────────────

def all_content_dirs() -> list[Path]:
    """Return every content folder in the repo, sorted by name."""
    dirs = [p for p in PROJECT_ROOT.iterdir() if p.is_dir() and CONTENT_DIR_RE.match(p.name)]
    day0 = PROJECT_ROOT / DAY0_DIR_NAME
    if day0.is_dir():
        dirs.append(day0)
    return sorted(dirs, key=lambda p: p.name)


def dirs_for_days(days: list[str] | None) -> list[Path]:
    """Return the content folders for the requested days (None = all days).

    Day "3" selects 3.1-tokenization, 3.2-jailbreaking, ... and day "0" selects
    day0-setup.
    """
    content_dirs = all_content_dirs()
    if days is None:
        return content_dirs

    selected: list[Path] = []
    for day in days:
        if day == "0":
            matches = [p for p in content_dirs if p.name == DAY0_DIR_NAME]
        else:
            matches = [p for p in content_dirs if p.name.startswith(f"{day}.")]
        if not matches:
            print(f"WARNING: no content folders found for day {day}", file=sys.stderr)
        selected += matches

    # A folder named on the command line twice should still only run once.
    return sorted(set(selected), key=lambda p: p.name)


def find_solutions(dirs: list[Path]) -> list[Path]:
    """Return the *_solution.py file in each folder, in folder order."""
    solutions: list[Path] = []
    for directory in dirs:
        solutions += sorted(directory.glob(f"*{SOLUTION_SUFFIX}"))
    return solutions


def rel(path: Path) -> str:
    """Path relative to the repo root, for readable output."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Generating and running one section
# ─────────────────────────────────────────────────────────────────────────────

def answers_path(solution: Path) -> Path:
    """section2_solution.py -> section2_answers.py, in the same folder."""
    return solution.with_name(solution.name[: -len(SOLUTION_SUFFIX)] + ANSWERS_SUFFIX)


def remove_stale_answers(dirs: list[Path]) -> None:
    """Delete answer files left behind by an interrupted earlier run.

    Normal runs clean up after themselves, so anything found here is debris
    from a crash or a SIGKILL -- and debris that contains full solutions.
    """
    for directory in dirs:
        for stale in sorted(directory.glob(f"*{ANSWERS_SUFFIX}")):
            stale.unlink()
            print(f"  Removed stale answers file: {rel(stale)}")


def generate_answers(solution: Path) -> Path:
    """Write the filled-in answers file next to its solution file.

    Reuses the same extraction the `--reference` build uses: keep the body of
    every `if "SOLUTION":` block, drop SKIP blocks, and replace each `test_*`
    definition with an import from the section's generated test module.
    """
    answers = answers_path(solution)
    test_file = solution.with_name(solution.name[: -len(SOLUTION_SUFFIX)] + TEST_SUFFIX)
    with open(solution, "r") as infile, open(answers, "w") as outfile:
        build_reference_py(infile, outfile, test_file.name)
    return answers


@dataclass
class Result:
    """Outcome of running one section's answers file."""

    solution: Path
    returncode: int
    seconds: float
    output: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_answers(solution: Path, answers: Path, timeout: int) -> Result:
    """Execute an answers file, streaming its output and capturing it too.

    The working directory is the section folder so that the answers file's
    `from sectionN_test import ...` and any relative data paths resolve exactly
    as they do for a participant working in that folder.
    """
    start = time.time()
    captured: list[str] = []
    timed_out = threading.Event()

    process = subprocess.Popen(
        # Run by bare filename with cwd set, so tracebacks show the short path.
        [sys.executable, answers.name],
        cwd=answers.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        # Without this the child block-buffers its stdout into our pipe and the
        # live stream arrives all at once at the end.
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    # Reading the pipe blocks, so the timeout needs its own timer rather than
    # process.wait(timeout=...), which would only be reached after the child
    # has already closed stdout.
    def kill_on_timeout() -> None:
        timed_out.set()
        process.kill()

    watchdog = threading.Timer(timeout, kill_on_timeout)
    watchdog.start()
    try:
        assert process.stdout is not None
        for line in process.stdout:
            # Indent the child's output so it is visually nested under the
            # section header we printed before starting it.
            print(f"    {line}", end="")
            captured.append(line)
        returncode = process.wait()
    finally:
        watchdog.cancel()

    if timed_out.is_set():
        message = f"TIMED OUT after {timeout}s (killed)\n"
        print(f"    {message}", end="")
        captured.append(message)

    return Result(
        solution=solution,
        returncode=returncode,
        seconds=time.time() - start,
        output="".join(captured),
        timed_out=timed_out.is_set(),
    )


def check_solution(solution: Path, timeout: int, remain: bool = False) -> Result:
    """Generate and run one section's answers file, deleting it afterwards.

    Pass remain=True to leave the generated file on disk for debugging.
    """
    answers = generate_answers(solution)
    try:
        return run_answers(solution, answers, timeout)
    finally:
        # Runs on success, on failure, and on KeyboardInterrupt, so the answers
        # file does not survive the run that created it -- unless the caller
        # explicitly asked to keep it.
        if not remain:
            answers.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[Result]) -> None:
    """Print the pass/fail table, then replay the tail of each failure."""
    failures = [r for r in results if not r.ok]

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {len(results) - len(failures)} passed, {len(failures)} failed")
    print(f"{'=' * 70}")
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        detail = "" if result.ok else f"  (exit {result.returncode})"
        print(f"  {status}  {rel(result.solution):<55} {result.seconds:6.1f}s{detail}")

    if not failures:
        return

    for result in failures:
        reason = "timed out" if result.timed_out else f"exited {result.returncode}"
        print(f"\n{'=' * 70}")
        print(f"  FAILED: {rel(result.solution)} ({reason})")
        print(f"{'=' * 70}")
        lines = result.output.splitlines()
        if len(lines) > FAILURE_TAIL_LINES:
            print(f"  ... {len(lines) - FAILURE_TAIL_LINES} earlier lines omitted ...")
        for line in lines[-FAILURE_TAIL_LINES:]:
            print(f"  {line}")

    print(f"\n{'=' * 70}")
    print(f"  {len(failures)} section(s) failed:")
    for result in failures:
        print(f"    {rel(result.solution)}")
    print(f"{'=' * 70}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill in every solution, run it, and report what fails.",
    )
    parser.add_argument(
        "--day",
        action="append",
        dest="days",
        metavar="N",
        help="Day to test, e.g. --day 3 runs every 3.x section. Repeatable.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test every section in the repo.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-section timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the sections that would run, then exit without running them.",
    )
    parser.add_argument(
        "--remain",
        action="store_true",
        help="Keep the generated *_answers.py files instead of deleting them, "
             "so a failing section can be edited and re-run by hand. They hold "
             "the full solutions, so avoid this on pods participants can reach.",
    )
    args = parser.parse_args()

    if bool(args.days) == bool(args.all):
        parser.error("give exactly one of --day N (repeatable) or --all")

    dirs = dirs_for_days(None if args.all else args.days)
    solutions = find_solutions(dirs)
    if not solutions:
        print("No solution files found for the requested days.", file=sys.stderr)
        return 1

    if args.list:
        print(f"Would test {len(solutions)} section(s):")
        for solution in solutions:
            print(f"  {rel(solution)}")
        return 0

    print(f"\n{'=' * 70}")
    print(f"  Testing {len(solutions)} section(s) from {PROJECT_ROOT}")
    print(f"{'=' * 70}")
    remove_stale_answers(dirs)

    if args.remain:
        print(
            "\n  --remain: generated answers files will be KEPT after each run.\n"
            "  They contain the full solutions. Delete them, or re-run without\n"
            "  --remain, before letting participants near this checkout."
        )

    results: list[Result] = []
    try:
        for index, solution in enumerate(solutions, start=1):
            print(f"\n--- [{index}/{len(solutions)}] {rel(solution)}")
            results.append(check_solution(solution, args.timeout, remain=args.remain))
    except KeyboardInterrupt:
        # check_solution's finally clause has already dealt with the answers
        # file for the section that was running; report what completed.
        print("\nInterrupted -- reporting the sections that finished.", file=sys.stderr)

    print_summary(results)
    if args.remain and results:
        print("\n  Answers files kept for debugging:")
        for result in results:
            kept = answers_path(result.solution)
            if kept.exists():
                print(f"    {rel(kept)}")
        print("  The next run sweeps these away before regenerating them.")

    return 1 if any(not r.ok for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
