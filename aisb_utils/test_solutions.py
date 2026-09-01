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
    python aisb_utils/test_solutions.py --day 3 --skip 3.3.5   # all but that exercise
    python aisb_utils/test_solutions.py --day 3 --skip 3.4     # all but that section

Every run starts with a CUDA preflight, and a day that needs a GPU stops there
if there is not a usable one (--no-gpu-check overrides).

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
from dataclasses import dataclass, field, replace
from pathlib import Path

import libcst as cst

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

# Days whose sections only call a hosted API: they need an OpenRouter key, not
# a GPU. Every other day loads local model weights, and without a usable GPU
# those sections either crash on the first .to("cuda") or silently fall back to
# CPU and grind until the timeout kills them -- which looks like a hang, not a
# missing GPU. Mirrors LIGHT_SETUP_DAYS in runpod_setup/deploy_runpod.py.
API_ONLY_DAYS = {"1", "2"}

# Run in a subprocess so the check cannot leave a CUDA context (and its several
# hundred MB of VRAM) behind in this process for the sections to trip over.
# nvidia-smi succeeding is not enough and neither is torch.cuda.is_available():
# a host with a faulty driver or UVM state reports a healthy device and only
# fails when a kernel actually launches, so this multiplies a real tensor --
# the same check 3.1-tokenization/setup/setup_pod.sh runs at provision time.
GPU_PREFLIGHT_SNIPPET = """
import sys
import torch

if not torch.cuda.is_available():
    sys.exit(f"torch {torch.__version__}: torch.cuda.is_available() is False")
x = torch.randn(1024, 1024, device="cuda")
(x @ x).sum().item()
props = torch.cuda.get_device_properties(0)
print(f"{props.name}, {props.total_memory / 1e9:.1f} GB, torch {torch.__version__}")
"""
GPU_PREFLIGHT_TIMEOUT = 180


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
# Skipping sections and exercises
# ─────────────────────────────────────────────────────────────────────────────

# --skip takes either a section ("3.3", the whole folder) or a single exercise
# ("3.3.5", one exercise inside it).
SKIP_ID_RE = re.compile(r"^\d+\.\d+(\.\d+)?$")

# Exercises are introduced by a "### Exercise D.S.E: Title" heading inside a
# top-level prose string, per the content structure conventions in CLAUDE.md.
# A trailing letter ("3.3.5a", an optional variant) belongs to its parent
# exercise, so skipping 3.3.5 skips 3.3.5a with it.
EXERCISE_HEADING_RE = re.compile(r"^###\s+Exercise\s+(\d+\.\d+\.\d+)", re.MULTILINE)

# Any "# " or "## " heading ends the exercise that was in effect: the prose has
# moved on to a new day, section, or the closing Summary. A "### " heading that
# is not an exercise (e.g. "### Section Name" in the learning objectives block)
# changes nothing.
CLOSING_HEADING_RE = re.compile(r"^#{1,2}\s+\S", re.MULTILINE)


def prose_text(node: cst.CSTNode) -> str | None:
    """The text of a top-level prose string statement, or None for code.

    Prose in a solution file is a bare triple-quoted string at module level,
    which parses as a statement line holding a single string expression.
    """
    if not isinstance(node, cst.SimpleStatementLine) or len(node.body) != 1:
        return None
    statement = node.body[0]
    if not isinstance(statement, cst.Expr) or not isinstance(statement.value, cst.SimpleString):
        return None
    value = statement.value.evaluated_value
    return value if isinstance(value, str) else None


def exercise_markers(text: str) -> list[str | None]:
    """Exercise-boundary markers in one prose block, in the order they appear.

    An exercise id means "everything after this belongs to that exercise";
    None means "the exercise in effect has ended".
    """
    markers: list[tuple[int, str | None]] = []
    markers += [(m.start(), m.group(1)) for m in EXERCISE_HEADING_RE.finditer(text)]
    markers += [(m.start(), None) for m in CLOSING_HEADING_RE.finditer(text)]
    return [marker for _, marker in sorted(markers, key=lambda pair: pair[0])]


def strip_exercises(source: str, skip: set[str]) -> tuple[str, set[str]]:
    """Remove the top-level statements belonging to the given exercise ids.

    An exercise owns its "### Exercise D.S.E" prose block and every statement
    after it, up to the next exercise heading or the next "#"/"##" heading --
    so the whole exercise goes: its prose, its functions, the calls that run
    them, and the imports of its tests.

    Returns the rewritten source and the subset of `skip` that actually
    matched, so the caller can flag ids that never applied to anything.

    Skipping an exercise whose functions a *later* exercise depends on will
    make the section fail with a NameError; skip a whole section instead.
    """
    module = cst.parse_module(source)
    kept: list[cst.BaseStatement] = []
    matched: set[str] = set()
    current: str | None = None

    for node in module.body:
        # A statement belongs to the exercise in effect when it starts, so a
        # prose block that opens an exercise belongs to that new exercise.
        owner = current
        text = prose_text(node)
        if text is not None:
            markers = exercise_markers(text)
            if markers:
                owner = markers[0]
                current = markers[-1]
        if owner in skip:
            matched.add(owner)
        else:
            kept.append(node)

    return module.with_changes(body=kept).code, matched


def section_id(solution: Path) -> str:
    """The "D.S" id of a section, from its folder name (3.3-guardrails -> 3.3)."""
    return solution.parent.name.split("-", 1)[0]


def drop_skipped_sections(solutions: list[Path], skip: set[str]) -> tuple[list[Path], set[str]]:
    """Filter out whole sections named by --skip; return the rest and what matched."""
    sections = {s for s in skip if s.count(".") == 1}
    kept = [s for s in solutions if section_id(s) not in sections]
    for solution in solutions:
        if section_id(solution) in sections:
            print(f"  Skipping section {rel(solution)} (--skip {section_id(solution)})")
    return kept, {section_id(s) for s in solutions} & sections


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


def generate_answers(solution: Path, skip: set[str] = frozenset()) -> tuple[Path, set[str]]:
    """Write the filled-in answers file next to its solution file.

    Reuses the same extraction the `--reference` build uses -- keep the body of
    every `if "SOLUTION":` block, drop SKIP blocks -- but with inline_tests=True,
    For Day 5, each `test_*` stays defined inline rather than being replaced by
    an import from the section's generated test module, so the run uses one
    process with a single copy of the shared TEST_FIXTURE setup (one loaded
    model) -- as a student's file does -- and identity-based checks behave the
    same as they do for a participant. Other days keep the import-based build.
    Any exercise ids in `skip` are then cut from the result.

    Returns the answers path and the skipped ids that matched this section.
    """
    answers = answers_path(solution)
    test_file = solution.with_name(solution.name[: -len(SOLUTION_SUFFIX)] + TEST_SUFFIX)
    # Scope the inline-tests build (single shared-fixture load) to Day 5 for now:
    # its sections compare object identity of a shared model, which needs one
    # model instance. Every other day keeps the import-based build unchanged, so
    # this cannot regress them. Widen this once a regression pass confirms it.
    day = solution.parent.name.split(".", 1)[0]
    inline = day == "5"
    with open(solution, "r") as infile, open(answers, "w") as outfile:
        build_reference_py(infile, outfile, test_file.name, inline_tests=inline)

    exercises = {s for s in skip if s.count(".") == 2}
    if not exercises:
        return answers, set()

    source, matched = strip_exercises(answers.read_text(), exercises)
    answers.write_text(source)
    for exercise in sorted(matched):
        print(f"  Skipping exercise {exercise} (--skip {exercise})")
    return answers, matched


@dataclass
class Result:
    """Outcome of running one section's answers file."""

    solution: Path
    returncode: int
    seconds: float
    output: str
    timed_out: bool = False
    # Exercise ids cut from this section by --skip, so the summary can say the
    # PASS was not over the whole section.
    skipped: tuple[str, ...] = field(default_factory=tuple)

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


def check_solution(
    solution: Path,
    timeout: int,
    remain: bool = False,
    skip: set[str] = frozenset(),
) -> Result:
    """Generate and run one section's answers file, deleting it afterwards.

    Pass remain=True to leave the generated file on disk for debugging, and
    skip={"3.3.5", ...} to cut those exercises before running.
    """
    answers, skipped = generate_answers(solution, skip)
    try:
        result = run_answers(solution, answers, timeout)
        return replace(result, skipped=tuple(sorted(skipped)))
    finally:
        # Runs on success, on failure, and on KeyboardInterrupt, so the answers
        # file does not survive the run that created it -- unless the caller
        # explicitly asked to keep it.
        if not remain:
            answers.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def gpu_preflight() -> tuple[bool, str]:
    """Check that a CUDA device exists and can actually run a kernel.

    Returns (ok, detail) instead of exiting, so the caller decides whether a
    missing GPU is fatal for the sections it is about to run.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", GPU_PREFLIGHT_SNIPPET],
            capture_output=True, text=True, timeout=GPU_PREFLIGHT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False, f"the check itself hung for {GPU_PREFLIGHT_TIMEOUT}s"
    if proc.returncode != 0:
        # sys.exit(str) puts the reason on stderr; an ImportError lands there too.
        detail = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
        return False, detail[-1] if detail else f"exit {proc.returncode}"
    return True, proc.stdout.strip()


def print_summary(results: list[Result]) -> None:
    """Print the pass/fail table, then replay the tail of each failure."""
    failures = [r for r in results if not r.ok]

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {len(results) - len(failures)} passed, {len(failures)} failed")
    print(f"{'=' * 70}")
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        detail = "" if result.ok else f"  (exit {result.returncode})"
        if result.skipped:
            detail += f"  (skipped {', '.join(result.skipped)})"
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
        "--no-gpu-check",
        action="store_true",
        help="Skip the CUDA preflight and run the sections anyway. Only useful "
             "for checking a GPU day's non-GPU parts on a machine without one.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        dest="skip",
        default=[],
        metavar="ID",
        help="Leave out a section (--skip 3.3) or a single exercise "
             "(--skip 3.3.5) and still count the rest as a pass. Repeatable. "
             "Use it for a known-failing stretch exercise you do not want "
             "failing the whole run; a skipped exercise whose code a later "
             "exercise needs will fail with a NameError, so skip the section "
             "in that case.",
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

    for skip_id in args.skip:
        if not SKIP_ID_RE.match(skip_id):
            parser.error(f"--skip expects a section or exercise id like 3.3 or 3.3.5, got {skip_id!r}")
    skip = set(args.skip)

    dirs = dirs_for_days(None if args.all else args.days)
    solutions = find_solutions(dirs)
    solutions, matched_skips = drop_skipped_sections(solutions, skip)
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

    # Fail fast on a GPU day with no usable GPU. Without this the run gets as
    # far as the first section that loads weights and then either dies with a
    # CUDA error 20 minutes in, or quietly runs on CPU until the timeout.
    needs_gpu = args.all or any(day not in API_ONLY_DAYS for day in args.days or [])
    if args.no_gpu_check:
        print("\n  GPU: check skipped (--no-gpu-check)")
    else:
        gpu_ok, gpu_detail = gpu_preflight()
        print(f"\n  GPU: {'OK -- ' + gpu_detail if gpu_ok else 'UNAVAILABLE -- ' + gpu_detail}")
        if needs_gpu and not gpu_ok:
            days = ", ".join(args.days) if args.days else "all"
            # The banner above goes to stdout; flush it so this block does not
            # overtake it when both streams are piped (as they are over SSH).
            sys.stdout.flush()
            print(
                f"\n  Day {days} needs a working GPU: its sections load model "
                f"weights locally.\n"
                "  Running them now would fail on the first CUDA call, or fall "
                "back to CPU\n  and grind until the per-section timeout. Fix the "
                "pod, or pass --no-gpu-check\n  to run them anyway.",
                file=sys.stderr,
            )
            return 1

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
            result = check_solution(solution, args.timeout, remain=args.remain, skip=skip)
            matched_skips |= set(result.skipped)
            results.append(result)
    except KeyboardInterrupt:
        # check_solution's finally clause has already dealt with the answers
        # file for the section that was running; report what completed.
        print("\nInterrupted -- reporting the sections that finished.", file=sys.stderr)

    print_summary(results)

    # A typo in --skip would otherwise pass silently and the exercise would run.
    unmatched = skip - matched_skips
    if unmatched:
        print(
            f"\n  WARNING: --skip {', '.join(sorted(unmatched))} matched nothing -- "
            "check the id, and that it belongs to a day that ran.",
            file=sys.stderr,
        )

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
